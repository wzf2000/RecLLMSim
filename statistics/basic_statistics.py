import os
import json
import tiktoken

from utils import SIM_DIR, HUMAN_DIR, HUMAN_DIR_V2

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

def get_sim_data(task: str | None = None, language: str = 'en') -> list[dict]:
    if language == 'zh' and task is not None and task in task_translation:
        task = task_translation[task]
    if task is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    elif task == 'travel planning':
        tasks = ['new travel planning', 'travel planning']
    else:
        tasks = [task]
    ret = []
    for task in tasks:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(SIM_DIR, task, file), 'r') as f:
                data = json.load(f)
            ret.append({
                'task': data['task_context'],
                'profile': data['preference']
            })
    return ret

def get_human_data(task: str | None = None) -> list[dict]:
    if task is not None and task in task_translation_reverse:
        task = task_translation_reverse[task]
    if task is None:
        tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    else:
        tasks = [task]
    ret = []

    def update(dir_name: str):
        users = os.listdir(dir_name)
        for user in users:
            for task in tasks:
                if not os.path.exists(os.path.join(dir_name, user, task)):
                    continue
                files = os.listdir(os.path.join(dir_name, user, task))
                files = [file for file in files if file.endswith('.json')]
                files.sort(key=lambda x: x.split('.')[0])
                for file in files:
                    with open(os.path.join(dir_name, user, task, file), 'r') as f:
                        data = json.load(f)
                    ret.append({
                        'task': data['task_context'],
                        'profile': f'{dir_name}/{user}'
                    })

    update(HUMAN_DIR)
    update(HUMAN_DIR_V2)
    return ret

if __name__ == '__main__':
    for task in [None, 'travel planning', 'recipe planning', 'preparing gifts', 'skills learning planning']:
        sim_data = get_sim_data(task, 'zh')
        human_data = get_human_data(task)
        if task is None:
            scenario = 'All'
        else:
            scenario = task.split()
            scenario = [e.capitalize() for e in scenario]
            scenario = ' '.join(scenario)
        sim_tasks = [e['task'] for e in sim_data]
        human_tasks = [e['task'] for e in human_data]
        sim_profiles = [e['profile'] for e in sim_data]
        human_profiles = [e['profile'] for e in human_data]
        sim_tasks = set(sim_tasks)
        human_tasks = set(human_tasks)
        sim_profiles = set(sim_profiles)
        human_profiles = set(human_profiles)
        print(f'--- {scenario} ---')
        print(f'#Tasks: {len(sim_tasks)} vs {len(human_tasks)}')
        print(f'#Profiles: {len(sim_profiles)} vs {len(human_profiles)}')
        while input('Continue? (y/n) ') != 'y':
            pass
