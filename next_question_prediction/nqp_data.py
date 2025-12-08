import os
import json
import jieba
import random
from loguru import logger
from sklearn.model_selection import train_test_split

from utils import get_profile, conv_format, HUMAN_DIR, HUMAN_DIR_V2, SIM_DIR, SIM_DIR_V2


gt_map = {
    '不够细致': 0,
    '不满足需求': 1,
    '不可用': 2,
    '不够多样': 3,
    '其它': 4
}

gt_map_reverse = {
    0: '不够细致',
    1: '不满足需求',
    2: '不可用',
    3: '不够多样',
    4: '其它'
}

def get_task_list(human: bool, task_name: str) -> list[str]:
    if human:
        return {
            'travel': ['旅行规划'],
            'gift': ['礼物准备'],
            'recipe': ['菜谱规划'],
            'skill': ['技能学习规划'],
            'all': ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
        }[task_name]
    else:
        return {
            'travel': ['new travel planning', 'travel planning', '旅行规划'],
            'gift': ['preparing gifts', '礼物准备'],
            'recipe': ['recipe planning', '菜谱规划'],
            'skill': ['skills learning planning', '技能学习规划'],
            'all': ['new travel planning', 'preparing gifts', 'recipe planning', 'skills learning planning', 'travel planning', '旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
        }[task_name]

def get_nqp_data(sample: bool = False, task_list: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    random.seed(42)
    data_list: list[dict] = []
    id_cnt = 0
    all_task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    if task_list is None:
        task_list = all_task_list

    def get_data_from_dir(dir_name: str):
        users = os.listdir(dir_name)
        users.sort()
        for user in users:
            for task in all_task_list:
                if not os.path.exists(os.path.join(dir_name, user, task)):
                    continue
                files = os.listdir(os.path.join(dir_name, user, task))
                files = [file for file in files if file.endswith('.json')]
                files.sort(key=lambda x: x.split('.')[0])
                for file in files:
                    with open(os.path.join(dir_name, user, task, file), 'r') as f:
                        data = json.load(f)
                    for i, utt in enumerate(data['history']):
                        if utt['role'] != 'user' or i == 0:
                            continue
                        if 'intent_label' not in data['history'][i]:
                            continue
                        nonlocal id_cnt
                        data_list.append({
                            'id': id_cnt,
                            'task': task,
                            'user': user,
                            'turn': i,
                            'file_path': os.path.join(dir_name, user, task, file),
                            'history': conv_format(data['history'][:i]),
                            'cut_history': ' '.join([turn['content_cut'] for turn in data['history'][:i]]),
                            'user_question': data['history'][i]['content'],
                            'origin_history': data['history'][:i],
                            'profile': get_profile(data['profile']),
                            'task_context': data['task_context'],
                            'intent': data['history'][i]['intent_label'],
                        })
                        id_cnt += 1

    get_data_from_dir(HUMAN_DIR)
    get_data_from_dir(HUMAN_DIR_V2)
    logger.info(f"Total data: {len(data_list)}")
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data = [item for item in train_data if item['task'] in task_list]
    test_data = [item for item in test_data if item['task'] in task_list]
    logger.info(f"Train data after filtering for {task_list}: {len(train_data)}")
    logger.info(f"Test data after filtering for {task_list}: {len(test_data)}")
    if sample:
        test_data = test_data[:10]
    return train_data, test_data

def get_nqp_data_sim(task_list: list[str] | None = None, sim_dir: str = SIM_DIR) -> list[dict]:
    data_list: list[dict] = []
    all_task_list = ['new travel planning', 'preparing gifts', 'recipe planning', 'skills learning planning', 'travel planning', '旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    if task_list is None:
        task_list = all_task_list
    for task in all_task_list:
        if not os.path.exists(os.path.join(sim_dir, task)):
            continue
        language = 'en' if task in ['new travel planning', 'preparing gifts', 'recipe planning', 'skills learning planning', 'travel planning'] else 'zh'
        content_field = 'content_zh' if language == 'en' else 'content'
        content_cut_field = 'content_zh_cut' if language == 'en' else 'content_cut'
        preference_field = 'preference_zh' if language == 'en' else 'preference'
        task_context_field = 'task_context_zh' if language == 'en' else 'task_context'
        files = os.listdir(os.path.join(sim_dir, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            try:
                with open(os.path.join(sim_dir, task, file), 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {os.path.join(sim_dir, task, file)}: {e}")
                raise e
            for i, utt in enumerate(data['history']):
                if utt['role'] != 'user' or i == 0:
                    continue
                if 'intent_label' not in data['history'][i]:
                    continue
                origin_history = [
                    {
                        'role': turn['role'],
                        'content': turn[content_field],
                        'content_cut': turn[content_cut_field],
                    }
                    for turn in data['history'][:i]
                ]
                data_list.append({
                    'task': task,
                    'turn': i,
                    'file_path': os.path.join(sim_dir, task, file),
                    'history': conv_format(data['history'][:i], content_field=content_field),
                    'cut_history': ' '.join([turn[content_cut_field] for turn in data['history'][:i]]),
                    'user_question': data['history'][i][content_field],
                    'origin_history': origin_history,
                    'profile': data[preference_field],
                    'task_context': data[task_context_field],
                    'intent': data['history'][i]['intent_label'],
                })
    logger.info(f"Total data: {len(data_list)}")
    data_list = [item for item in data_list if item['task'] in task_list]
    logger.info(f"Data for {task_list}: {len(data_list)}")
    return data_list

def get_nqp_data_sim_rewritten(task_list: list[str] | None = None) -> list[dict]:
    sim_dir = SIM_DIR_V2
    data_list: list[dict] = []
    all_task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    if task_list is None:
        task_list = all_task_list
    for task in all_task_list:
        if not os.path.exists(os.path.join(sim_dir, task)):
            continue
        content_field = 'content_rewritten'
        content_cut_field = 'content_rewritten_cut'
        preference_field = 'preference'
        task_context_field = 'task_context'
        files = os.listdir(os.path.join(sim_dir, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(sim_dir, task, file), 'r') as f:
                data = json.load(f)
            if content_field not in data['history'][0]:
                continue
            for i, utt in enumerate(data['history']):
                if utt['role'] == 'user':
                    data['history'][i][content_cut_field] = ' '.join(jieba.cut(data['history'][i][content_field]))
                if utt['role'] != 'user' or i == 0:
                    continue
                if 'intent_label' not in data['history'][i]:
                    continue
                origin_history = [
                    {
                        'role': turn['role'],
                        'content': turn[content_field] if turn['role'] == 'user' else turn['content'],
                        'content_cut': turn[content_cut_field] if turn['role'] == 'user' else turn['content_cut'],
                    }
                    for turn in data['history'][:i]
                ]
                data_list.append({
                    'task': task,
                    'file_path': os.path.join(sim_dir, task, file),
                    'history': conv_format(data['history'][:i], content_field={
                        'user': content_field,
                        'assistant': 'content'
                    }),
                    'cut_history': ' '.join([turn[content_cut_field] if turn['role'] == 'user' else turn['content_cut'] for turn in data['history'][:i]]),
                    'user_question': data['history'][i][content_field],
                    'origin_history': origin_history,
                    'profile': data[preference_field],
                    'task_context': data[task_context_field],
                    'intent': data['history'][i]['intent_label'],
                })
    logger.info(f"Total data: {len(data_list)}")
    data_list = [item for item in data_list if item['task'] in task_list]
    logger.info(f"Data for {task_list}: {len(data_list)}")
    return data_list
