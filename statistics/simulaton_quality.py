import os
import json

from utils import SIM_DIR

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

rating_keys = ['Preference Alignment', 'Role-Playing Completeness']

def get_sim_data(task: str | None = None) -> dict[int, dict[str, list[str]]]:
    if task is not None and task in task_translation:
        task = task_translation[task]
    if task is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    elif task == 'travel planning':
        tasks = ['new travel planning', 'travel planning']
    else:
        tasks = [task]
    quality: dict[int, dict[str, list[str]]] = {}
    for task in tasks:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(SIM_DIR, task, file), 'r') as f:
                data = json.load(f)
            turns = len(data['history']) // 2
            if turns not in quality:
                quality[turns] = {}
            for key in rating_keys:
                if key not in quality[turns]:
                    quality[turns][key] = []
                quality[turns][key].append(data['rating'][key])
    return quality

if __name__ == '__main__':
    quality_data = get_sim_data()
    # sort by turns
    quality_data = dict(sorted(quality_data.items(), key=lambda x: x[0]))
    # output markdown format table
    # head
    print(f'| Turns | # Conv. | {" | ".join(rating_keys)} |')
    print('| --- | --- | --- | --- |') 
    # body
    for turns, data in quality_data.items():
        print(f'| {turns} | {len(data[rating_keys[0]])} | {" | ".join([f"{sum(data[key]) / len(data[key]):.4f}" for key in rating_keys])} |')
