import os
import json
from typing import Iterable

from utils import SIM_DIR, HUMAN_DIR, LABEL_FILE

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

def get_sim_data(task: str | None = None) -> list[dict[str, list[str]]]:
    if task is not None and task in task_translation:
        task = task_translation[task]
    if task is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    elif task == 'travel planning':
        tasks = ['new travel planning', 'travel planning']
    else:
        tasks = [task]
    profiles: list[dict[str, list[str]]] = []
    for task in tasks:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(SIM_DIR, task, file), 'r') as f:
                data = json.load(f)
            profile = data['profile_mapped']

            output_profile = {}
            for attr in profile:
                if not isinstance(profile[attr], list):
                    continue
                output_profile[attr] = [item['zh'] for item in profile[attr]]
            profiles.append(output_profile)
    return profiles

def get_human_data(task: str | None = None) -> list[dict[str, list[str]]]:
    if task is not None and task in task_translation_reverse:
        task = task_translation_reverse[task]
    if task is None:
        tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    else:
        tasks = [task]
    profiles: list[dict[str, list[str]]] = []
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
                profile = data['profile']
                output_profile = {}
                for attr in profile:
                    if not isinstance(profile[attr], list):
                        continue
                    output_profile[attr] = profile[attr]
                profiles.append(output_profile)
    return profiles

def average_length(data: Iterable[list]):
    return sum(len(conv) for conv in data) / len(data)

def format_attr(key: str) -> str:
    return key.split('and')[0].strip().lower().replace(' ', '_')

def read_labels() -> dict[str, dict[str, str]]:
    with open(LABEL_FILE, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    sim_data = get_sim_data()
    human_data = get_human_data()
    attributes = sim_data[0].keys()
    labels = read_labels()
    for attr in attributes:
        label_size = len(labels[format_attr(attr)])
        sim_attr = [profile[attr] for profile in sim_data]
        human_attr = [profile[format_attr(attr)] for profile in human_data]
        print(f'{attr} & {label_size} & {average_length(sim_attr):.2f} & {average_length(human_attr):.2f} \\\\')
