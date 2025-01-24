import os
import json
import jieba
import numpy as np
from enum import Enum

SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLM_agent_user')
HUMAN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real_human_user')
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

def format_history(history: list[dict[str, str]], content_field: str, model_type: ModelType, cut: bool) -> str | list[dict[str, str]]:
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

def get_sim_data(item: str, language: str = 'en', task: str | None = None, model_type: ModelType = ModelType.LLM, filtered: bool = False) -> tuple[list[str | list[dict[str, str]]], list[set[str]]]:
    if language == 'zh' and task is not None and task in task_translation:
        task = task_translation[task]
    if task is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    elif task == 'travel planning':
        tasks = ['new travel planning', 'travel planning']
    else:
        tasks = [task]
    labels: list[set[str]] = []
    X: list[str | list[dict[str, str]]] = []
    for task in tasks:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(SIM_DIR, task, file), 'r') as f:
                data = json.load(f)
            label = [ele[language] for ele in data['profile_mapped'][item]]
            label = set(label)
            if filtered:
                if len(label) == 1 and ('其他' in label or 'Others' in label):
                    continue
                if language == 'zh' and '其他' in label:
                    label.remove('其他')
                if language == 'en' and 'Others' in label:
                    label.remove('Others')
            labels.append(label)
            text = format_history(data['history'], 'content' if language == 'en' else 'content_zh', model_type, language == 'zh')
            X.append(text)
    return X, labels

def get_human_data(item: str, task: str | None = None, model_type: ModelType = ModelType.LLM) -> tuple[list[str | list[dict[str, str]]], list[set[str]]]:
    if task is not None and task in task_translation_reverse:
        task = task_translation_reverse[task]
    if task is None:
        tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    else:
        tasks = [task]
    item = item.split('and')[0].strip().lower().replace(' ', '_')
    labels: list[set[str]] = []
    X: list[str | list[dict[str, str]]] = []
    users = os.listdir(HUMAN_DIR)
    for user in users:
        for task in tasks:
            if not os.path.exists(os.path.join(HUMAN_DIR, user, task)):
                continue
            files = os.listdir(os.path.join(HUMAN_DIR, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(HUMAN_DIR, user, task, file), 'r') as f:
                    data = json.load(f)
                label = data['profile'][item]
                label = set(label)
                labels.append(label)
                text = format_history(data['history'], 'content', model_type, True)
                X.append(text)
    return X, labels

def get_human_intent_data(model_type: ModelType = ModelType.LLM):
    tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    labels: list[str] = []
    X: list[str | list[dict[str, str]]] = []
    users = os.listdir(HUMAN_DIR)
    for user in users:
        for task in tasks:
            if not os.path.exists(os.path.join(HUMAN_DIR, user, task)):
                continue
            files = os.listdir(os.path.join(HUMAN_DIR, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(HUMAN_DIR, user, task, file), 'r') as f:
                    data = json.load(f)
                for i, (utt1, utt2) in enumerate(zip(data['history'][:-1], data['history'][1:])):
                    if utt1['role'] != 'assistant' or utt2['role'] != 'user':
                        continue
                    label = utt2['intent'][0]
                    text = format_history(data['history'][:i + 2], 'content', model_type, True)
                    X.append(text)
                    labels.append(label)
    return X, np.array(labels)
