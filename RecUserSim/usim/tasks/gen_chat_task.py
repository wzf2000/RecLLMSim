import os
import json
import random
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from usim import PREFERENCE, TASK_CONTEXT, POOL_PATH, DATA_PATH
from usim.tasks.base_task import Task
from usim.chat import multi_chat, single_turn_chat
from usim.utils import to_int_list
from usim.utils.preferences import get_all_preference
from usim.utils.prompts import get_all_prompts

class GenChatTask(Task):
    @staticmethod
    def parse_task_args(parser):
        parser.add_argument('-c', '--config', default='config/gen_chat_en.json', type=str, help='The path of config file')
        parser.add_argument('--task_type', required=True, type=str, choices=[
            'all',
            '旅行规划', '技能学习规划', '餐厅选择', '礼物准备', '菜谱规划',
            "travel planning", "skills learning planning", "choosing restaurant", "preparing gifts", "recipe planning", "new travel planning"
        ])
        parser.add_argument('--context_ids', default=[], type=to_int_list)
        parser.add_argument('--test', action='store_true', help='Whether to test the task')
        parser.add_argument('--output_dir', default='output', type=str, help='The output directory')
        parser.add_argument('--post_query', action='store_true', help='Whether to test the task after conversation')
        parser.add_argument('--preferences', default=-1, type=int, help='The number of preferences to generate')
        parser.add_argument('--max_turn', default=20, type=int, help='The maximum number of turns')
        parser.add_argument('--sample', type=int, default=-1, help='Number of samples to run, -1 means all')
        return parser

    @staticmethod
    def _generate_prompt(prompt: str, preference: str = None, task_context: str = None) -> str:
        if preference is not None:
            prompt = prompt.replace(PREFERENCE, preference)
        if task_context is not None:
            prompt = prompt.replace(TASK_CONTEXT, task_context)
        return prompt

    def run(self, config: str, task_type: str, context_ids: list[int], test: bool, output_dir: str, post_query: bool, preferences: int, max_turn: int, sample: int, *args, **kwargs):
        self.task_type = task_type
        self.max_turn = max_turn
        with open(config, 'r') as f:
            self.config = json.load(f)

        preference_pool = get_all_preference(os.path.join(POOL_PATH, self.config['preference_pool']))
        if preferences < 0:
            preferences = len(preference_pool)
        task_context_list = get_all_prompts(os.path.join(DATA_PATH, self.config['task_context']), 'sheet1')
        context_task_map = {
            context: task
            for task, contexts in task_context_list.items() for context in contexts
        }
        if self.task_type != 'all':
            task_context_list = task_context_list[self.task_type]
        else:
            task_context_list = sum(task_context_list.values(), [])

        if len(context_ids) == 1 and context_ids[0] == -1:
            context_ids = list(range(len(task_context_list)))
        if len(context_ids) == 0:
            # print the help message
            print('Please select the context ids you want to run:')
            for i, task_context in enumerate(task_context_list):
                print(f'{i}. {task_context}')
            print('preference pool size is', len(preference_pool))
        else:
            if test:
                # random select one preference and one context id to test
                context_id = random.choice(context_ids)
                task_context = task_context_list[context_id]
                preference = random.choice(preference_pool)
                ret_json = self.single_run(preference, task_context)
                logger.debug(ret_json)
            else:
                # check output dir exists
                output_dir = os.path.join(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                enumerate_preference_pool = list(enumerate(preference_pool))
                if preferences < len(preference_pool):
                    random.seed(2024)
                    enumerate_preference_pool = random.sample(enumerate_preference_pool, preferences)
                task_contexts = [task_context_list[context_id] for context_id in context_ids]

                def process_single(tid: int, pid: int, cnt: int = 0):
                    task_context = task_contexts[tid]
                    task = context_task_map[task_context]
                    preference = preference_pool[pid]
                    if os.path.exists(os.path.join(output_dir, task, f'{pid}_{context_ids[tid]}.json')):
                        return
                    try:
                        ret_json = self.single_run(preference, task_context, post_query=post_query)
                        if ret_json is None:
                            logger.critical(f'[task context id = {context_ids[tid]}] Error occured when preference id = {pid}')
                            raise Exception('Run Error')
                        os.makedirs(os.path.join(output_dir, task), exist_ok=True)
                        with open(os.path.join(output_dir, task, f'{pid}_{context_ids[tid]}.json'), 'w') as f:
                            json.dump(ret_json, f, ensure_ascii=False, indent=4)
                        logger.debug(f'[task context id = {context_ids[tid]}, preference id = {pid}] Finish')
                    except Exception as e:
                        logger.critical(f'[task context id = {context_ids[tid]}] Exception occured on {cnt + 1}\'s try when preference id = {pid}: {e}')
                        process_single(tid, pid, cnt + 1)

                with ThreadPoolExecutor(max_workers=32) as executor:
                    futures = {}
                    if sample < 0:
                        logger.success(f'Start processing {len(context_ids) * len(enumerate_preference_pool)} tasks')
                        for tid in range(len(task_contexts)):
                            for pid, _ in enumerate_preference_pool:
                                futures[executor.submit(process_single, tid, pid)] = (tid, pid)
                    else:
                        logger.success(f'Start processing {sample} tasks')
                        all_tasks = []
                        for tid in range(len(task_contexts)):
                            for pid, _ in enumerate_preference_pool:
                                all_tasks.append((tid, pid))
                        random.seed(2024)
                        sampled_tasks = random.sample(all_tasks, sample)
                        for tid, pid in sampled_tasks:
                            futures[executor.submit(process_single, tid, pid)] = (tid, pid)
                    for future in tqdm(as_completed(futures), total=len(futures), desc='Processing'):
                        tid, pid = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.critical(f'[task context id = {context_ids[tid]}] Exception occured when preference id = {pid}: {e}')

    @logger.catch
    def single_run(self, preference: str, task_context: str, post_query: bool = False):
        def process1(x: str, turn: int) -> str:
            return x

        def process2(x: str, turn: int) -> str:
            if turn < self.max_turn - 3:
                return x + '\n' + self.config['end_prompt']
            else:
                # in the last 3 turns, remind the user to end the conversation
                return x + '\n' + self.config['turn_reminder'].format(
                    turn=turn + 1,
                    max_turn=self.max_turn
                )

        def ending(x: str) -> bool:
            for token in self.config['end_check']:
                if token.lower() not in x.lower():
                    return False
            return True

        prompt1_list = self.config['prompts']['1']
        prompt2_list = self.config['prompts']['2']
        prompt2_list = [self._generate_prompt(prompt, preference, task_context) for prompt in prompt2_list]

        if 'model' in self.config:
            # generate seed by hashing preference and task_context
            random.seed(hash(preference + task_context) % (2 ** 32))
            if isinstance(self.config['model']['1'], str):
                model1 = self.config['model']['1']
                logger.info(f'Using model1: {model1}')
            else:
                assert isinstance(self.config['model']['1'], list)
                model1 = random.choice(self.config['model']['1'])
            if isinstance(self.config['model']['2'], str):
                model2 = self.config['model']['2']
            else:
                assert isinstance(self.config['model']['2'], list)
                model2 = random.choice(self.config['model']['2'])
                logger.info(f'Using model2: {model2}')
            chatbot1, chatbot2, ended = multi_chat(prompt1_list, prompt2_list, ending=ending, process1=process1, process2=process2, model1=model1, model2=model2, max_turn=self.max_turn)
        else:
            chatbot1, chatbot2, ended = multi_chat(prompt1_list, prompt2_list, ending=ending, process1=process1, process2=process2, max_turn=self.max_turn)
        if not ended:
            logger.warning('Conversation did not end properly')
            return None

        ret = {
            'preference': preference,
            'task_context': task_context,
            'history': chatbot1.history[len(prompt1_list) * 2:],
            'model1': chatbot1.model,
            'model2': chatbot2.model,
        }
        if not post_query:
            return ret

        ret['post_questions'] = {
            'assistant': [],
            'user': []
        }

        logger.info('Stage: Test After Conversation')
        if 'end_test' not in self.config:
            logger.debug('No end test')
            return
        final_history1 = chatbot1.history
        for test_utt in self.config['end_test']['1']:
            chatbot1.history = final_history1
            logger.debug(f'Test for ChatGPT1: {test_utt}')
            response = single_turn_chat(chatbot1, 1, test_utt)
            ret['post_questions']['assistant'].append({
                'query': test_utt,
                'response': response
            })
        final_history2 = chatbot2.history
        for test_utt in self.config['end_test']['2']:
            chatbot2.history = final_history2
            logger.debug(f'Test for ChatGPT2: {test_utt}')
            response = single_turn_chat(chatbot2, 2, test_utt)
            ret['post_questions']['user'].append({
                'query': test_utt,
                'response': response
            })
        return ret
