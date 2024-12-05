import os
import json
from tqdm import tqdm

from strategy_cls import multi_cls, conv_format
from basic_info import SIM_DIR, HUMAN_DIR

def cls_sim(version: int, model: str, strategy_name: str, compare_strategy_name: str | None, sample: bool = False):
    dir_name = SIM_DIR
    task_list = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']

    if compare_strategy_name is None:
        right = {}

    for task in task_list:
        files = os.listdir(os.path.join(dir_name, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        if sample:
            files = files[:30]
        for file in tqdm(files, desc=task):
            with open(os.path.join(dir_name, task, file), 'r') as f:
                data = json.load(f)
            if strategy_name in data:
                if compare_strategy_name is None:
                    for key in data[strategy_name]['final']:
                        if key not in right:
                            right[key] = []
                        right[key].append(data[strategy_name]['final'][key] == data['strategy']['final'][key])
                continue
            try:
                data[strategy_name]= multi_cls(conv_format(data['history']), model, version, False)
                if compare_strategy_name is None:
                    for key in data[strategy_name]['final']:
                        if key not in right:
                            right[key] = []
                        right[key].append(data[strategy_name]['final'][key] == data[compare_strategy_name]['final'][key])
                with open(os.path.join(dir_name, task, file), 'w') as fw:
                    json.dump(data, fw, ensure_ascii=False, indent=4)
            except Exception as e:
                print(e)
                print(type(e))
                print(os.path.join(dir_name, task, file))

    if compare_strategy_name is None:
        for key in right:
            print(f'{key}: {sum(right[key]) / len(right[key])}')

def cls_human(version: int, model: str, strategy_name: str, compare_strategy_name: str | None):
    dir_name = HUMAN_DIR
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']

    if compare_strategy_name is None:
        right = {}

    for user in tqdm(os.listdir(dir_name)):
        for task in task_list:
            if not os.path.exists(os.path.join(dir_name, user, task)):
                continue
            files = os.listdir(os.path.join(dir_name, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(dir_name, user, task, file), 'r') as f:
                    data = json.load(f)
                if strategy_name in data:
                    if compare_strategy_name is None:
                        for key in data[strategy_name]['final']:
                            if key not in right:
                                right[key] = []
                            right[key].append(data[strategy_name]['final'][key] == data[compare_strategy_name]['final'][key])
                    continue
                try:
                    data[strategy_name]= multi_cls(conv_format(data['history']), model, version, True)
                    if compare_strategy_name is None:
                        for key in data[strategy_name]['final']:
                            if key not in right:
                                right[key] = []
                            right[key].append(data[strategy_name]['final'][key] == data[compare_strategy_name]['final'][key])
                    with open(os.path.join(dir_name, user, task, file), 'w') as fw:
                        json.dump(data, fw, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(e)
                    print(type(e))
                    print(os.path.join(dir_name, user, task, file))

    if compare_strategy_name is None:
        for key in right:
            print(f'{key}: {sum(right[key]) / len(right[key])}')

if __name__ == '__main__':
    cls_sim(1, 'gpt-4o-2024-11-20', 'strategy_1120', 'strategy', True)
    cls_sim(2, 'gpt-4o-2024-11-20', 'strategy_V2_1120', 'strategy_V2', True)
    cls_human(1, 'gpt-4o-2024-11-20', 'strategy_1120', 'strategy')
    cls_human(2, 'gpt-4o-2024-11-20', 'strategy_V2_1120', 'strategy_V2')
