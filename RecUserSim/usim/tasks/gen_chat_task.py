import os
import json
import random
from loguru import logger

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
            '旅行规划', '技能学习规划', '餐厅选择', '礼物准备', '菜谱规划',
            "travel planning", "skills learning planning", "choosing restaurant", "preparing gifts", "recipe planning", "new travel planning"
        ])
        parser.add_argument('--context_ids', default=[], type=to_int_list)
        parser.add_argument('--test', action='store_true', help='Whether to test the task')
        parser.add_argument('--output_dir', default='output', type=str, help='The output directory')
        parser.add_argument('--env_id', default='0', type=str, help='The environment id')
        parser.add_argument('--post_query',action='store_true', help='Whether to test the task after conversation')
        parser.add_argument('--preferences', default=-1, type=int, help='The number of preferences to generate')
        return parser
    
    @staticmethod
    def _generate_prompt(prompt: str, preference: str = None, task_context: str = None) -> str:
        if preference is not None:
            prompt = prompt.replace(PREFERENCE, preference)
        if task_context is not None:
            prompt = prompt.replace(TASK_CONTEXT, task_context)
        return prompt
    
    def run(self, config: str, task_type: str, context_ids: list[int], test: bool, output_dir: str, env_id: str, post_query: bool, preferences: int, *args, **kwargs):
        self.task_type = task_type
        with open(config, 'r') as f:
            self.config = json.load(f)

        preference_pool = get_all_preference(os.path.join(POOL_PATH, self.config['preference_pool']))
        if preferences < 0:
            preferences = len(preference_pool)
        task_context_list = get_all_prompts(os.path.join(DATA_PATH, self.config['task_context']), 'sheet1')[self.task_type]

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
                output_dir = output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_dir = os.path.join(output_dir, f"{self.task_type}_{env_id}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                enumerate_preference_pool = list(enumerate(preference_pool))
                if preferences < len(preference_pool):
                    random.seed(2024)
                    enumerate_preference_pool = random.sample(enumerate_preference_pool, preferences)
                task_contexts = [task_context_list[context_id] for context_id in context_ids]
                cnt = 0
                while cnt < len(task_contexts) * len(enumerate_preference_pool):
                    cnt = 0
                    for tid, task_context in enumerate(task_contexts):
                        for pid, preference in enumerate_preference_pool:
                            if os.path.exists(os.path.join(output_dir, f'{pid}_{context_ids[tid]}.json')):
                                cnt += 1
                                logger.warning(f'[task context id = {context_ids[tid]}] {pid}_{context_ids[tid]}.json exists')
                                continue
                            ret_json = self.single_run(preference, task_context, post_query=post_query)
                            if ret_json is None:
                                logger.critical(f'[task context id = {context_ids[tid]}] Error occured when preference id = {pid}')
                                continue
                            with open(os.path.join(output_dir, f'{pid}_{context_ids[tid]}.json'), 'w') as f:
                                json.dump(ret_json, f, ensure_ascii=False, indent=4)
                            cnt += 1
                            logger.success(f'[task context id = {context_ids[tid]}, preference id = {pid}] Finish {cnt}/{len(task_contexts) * len(enumerate_preference_pool)}')
                        logger.success(f'Finish {tid + 1}/{len(task_contexts)}')

    @logger.catch
    def single_run(self, preference: str, task_context: str, post_query: bool = False):
        process1 = lambda x: x
        process2 = lambda x: x + '\n' + self.config['end_prompt']
        # process2 = lambda x: x
        def ending(x):
            for token in self.config['end_check']:
                if token.lower() not in x.lower():
                    return False
            return True

        prompt1_list = self.config['prompts']['1']
        prompt2_list = self.config['prompts']['2']
        prompt2_list = [self._generate_prompt(prompt, preference, task_context) for prompt in prompt2_list]

        if 'engine' in self.config:
            chatbot1, chatbot2 = multi_chat(prompt1_list, prompt2_list, ending=ending, process1=process1, process2=process2, engine1=self.config['engine']['1'], engine2=self.config['engine']['2'])
        else:
            chatbot1, chatbot2 = multi_chat(prompt1_list, prompt2_list, ending=ending, process1=process1, process2=process2)

        ret = {}
        ret['preference'] = preference
        ret['task_context'] = task_context
        ret['history'] = chatbot1.history[len(prompt1_list) * 2:]
        if not post_query:
            return ret
        ret['post_questions'] = {
            'assistant': [],
            'user': []
        }

        logger.info('Stage: Test After Conversation')
        if 'end_test' not in self.config:
            logger.info('No end test')
            return
        final_history1 = chatbot1.history
        for test_utt in self.config['end_test']['1']:
            chatbot1.history = final_history1
            logger.info(f'Test for ChatGPT1: {test_utt}')
            response = single_turn_chat(chatbot1, 1, test_utt)
            ret['post_questions']['assistant'].append({
                'query': test_utt,
                'response': response
            })
        final_history2 = chatbot2.history
        for test_utt in self.config['end_test']['2']:
            chatbot2.history = final_history2
            logger.info(f'Test for ChatGPT2: {test_utt}')
            response = single_turn_chat(chatbot2, 2, test_utt)
            ret['post_questions']['user'].append({
                'query': test_utt,
                'response': response
            })
        return ret
    