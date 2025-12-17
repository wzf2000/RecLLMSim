import os
import json
import jieba
from enum import Enum

SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLM_agent_user')
SIM_DIR_V2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLM_agent_user_V2')
HUMAN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real_human_user')
HUMAN_DIR_V2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'human_exp_V2')
LABEL_FILE = os.path.join(os.path.dirname(__file__), 'desc_translated.json')

task_translation = {
    '旅行规划': 'travel planning',
    '礼物准备': 'preparing gifts',
    '菜谱规划': 'recipe planning',
    '技能学习规划': 'skills learning planning'
}

task_translation_reverse = {
    'travel planning': '旅行规划',
    'preparing gifts': '礼物准备',
    'recipe planning': '菜谱规划',
    'skills learning planning': '技能学习规划'
}

item_translation = {
    'Personality': '性格',
    'Daily Interests and Hobbies': '日常兴趣爱好',
    'Travel Habits': '旅行习惯',
    'Dining Preferences': '饮食偏好',
    'Spending Habits': '消费习惯',
    'Other Aspects': '其他方面'
}

class ModelType(Enum):
    LLM = 'large language model'
    ML = 'machine learning'
    LM = 'language model'
    HUMAN = 'human'

def format_history(history: list[dict[str, str]], content_field: str, model_type: ModelType, cut: bool, only: str | None = None) -> str | list[dict[str, str]]:
    if only is not None:
        assert only in ['user', 'assistant'], f'Invalid value for only: {only}'
        history = [utt for utt in history if utt['role'] == only]
        ret = '\n\n'.join([utt[content_field] for utt in history])
        if model_type == ModelType.ML and cut:
            ret = ' '.join(jieba.lcut(ret))
        return ret
    if model_type == ModelType.HUMAN:
        return [{
            'role': utt['role'],
            'content': utt['content']
        } for utt in history]
    text = ''
    for utt in history:
        if model_type == ModelType.LLM:
            text += f"{utt['role']}: {utt[content_field]}\n\n"
        else:
            text += utt[content_field]
    if model_type == ModelType.ML and cut:
        text = ' '.join(jieba.lcut(text))
    return text

def get_sim_data(task: str | None = None, model_type: ModelType = ModelType.LLM, sim_dir: str = SIM_DIR, rewritten: bool = False) -> tuple[list[str | list[dict[str, str]]], list[bool]]:
    if task is not None and task in task_translation:
        task = task_translation[task]
    if task is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning', '旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    elif task == 'travel planning':
        tasks = ['new travel planning', 'travel planning', '旅行规划']
    else:
        tasks = [task, task_translation_reverse.get(task, task)]
    labels: list[bool] = []
    X: list[str | list[dict[str, str]]] = []
    for task in tasks:
        if not os.path.exists(os.path.join(sim_dir, task)):
            continue
        files = os.listdir(os.path.join(sim_dir, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(sim_dir, task, file), 'r') as f:
                data = json.load(f)
            if rewritten:
                if 'content_rewritten' not in data['history'][0]:
                    continue
                else:
                    content_field = 'content_rewritten'
            else:
                content_field = 'content' if task in ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划'] else 'content_zh'
            labels.append(True)
            text = format_history(data['history'], content_field, model_type, True, only='user')
            X.append(text)
    return X, labels

def get_human_data(task: str | None = None, model_type: ModelType = ModelType.LLM) -> tuple[list[str | list[dict[str, str]]], list[bool]]:
    if task is not None and task in task_translation_reverse:
        task = task_translation_reverse[task]
    if task is None:
        tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    else:
        tasks = [task]

    labels: list[bool] = []
    X: list[str | list[dict[str, str]]] = []

    def update_data(user_dir: str):
        users = os.listdir(user_dir)
        users.sort(key=lambda x: int(x.split('_')[-1]))
        for user in users:
            for task in tasks:
                if not os.path.exists(os.path.join(user_dir, user, task)):
                    continue
                files = os.listdir(os.path.join(user_dir, user, task))
                files = [file for file in files if file.endswith('.json')]
                files.sort(key=lambda x: x.split('.')[0])
                for file in files:
                    with open(os.path.join(user_dir, user, task, file), 'r') as f:
                        data = json.load(f)
                    labels.append(False)
                    text = format_history(data['history'], 'content', model_type, True, only='user')
                    X.append(text)

    update_data(HUMAN_DIR)
    update_data(HUMAN_DIR_V2)
    return X, labels

def process_profile(profile: dict) -> dict:
    ret = {}
    for key in profile:
        transformed_key = key.split(' and ')[0].lower().replace(' ', '_')
        if isinstance(profile[key], dict):
            ret[transformed_key] = profile[key]['zh']
        elif isinstance(profile[key], list):
            ret[transformed_key] = [item['zh'] if isinstance(item, dict) and 'zh' in item else item for item in profile[key]]
        else:
            raise ValueError(f'Unexpected profile item type: {type(profile[key])}')
    return ret

def process_history(history: list[dict], rewritten: bool = False) -> list[dict]:
    ret = []
    for turn in history:
        role = turn['role']
        if rewritten and 'content_rewritten' in turn:
            content = turn['content_rewritten']
        else:
            content = turn['content_zh'] if 'content_zh' in turn else turn['content']
        ret.append({
            'role': role,
            'content': content
        })
    return ret

def get_sim_data_dict(dir_path: str = SIM_DIR, rewritten: bool = False) -> list[dict]:
    tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning', '旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    data: list[dict] = []
    for task in tasks:
        if not os.path.exists(os.path.join(dir_path, task)):
            continue
        language = 'zh' if task in ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划'] else 'en'
        files = os.listdir(os.path.join(dir_path, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(dir_path, task, file), 'r') as f:
                item = json.load(f)
                data.append({
                    'history': process_history(item['history'], rewritten),
                    'task_background': item['task_context_zh'] if language == 'en' else item['task_context'],
                    'file_path': os.path.join(dir_path, task, file),
                })
    return data

def get_human_data_dict() -> list[dict]:
    tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    data: list[dict] = []

    def update_data(user_dir: str):
        users = os.listdir(user_dir)
        users.sort(key=lambda x: int(x.split('_')[-1]))
        for user in users:
            for task in tasks:
                if not os.path.exists(os.path.join(user_dir, user, task)):
                    continue
                files = os.listdir(os.path.join(user_dir, user, task))
                files = [file for file in files if file.endswith('.json')]
                files.sort(key=lambda x: x.split('.')[0])
                for file in files:
                    with open(os.path.join(user_dir, user, task, file), 'r') as f:
                        item = json.load(f)
                        data.append({
                            'history': item['history'],
                            'task_background': item['task_context'],
                            'file_path': os.path.join(user_dir, user, task, file),
                        })

    update_data(HUMAN_DIR)
    update_data(HUMAN_DIR_V2)
    return data
