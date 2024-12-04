import os
import json
from tqdm import tqdm

from strategy_cls import multi_cls, conv_format
from basic_info import SIM_DIR, HUMAN_DIR

def cls_sim(version: int):
    dir_name = SIM_DIR
    task_list = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    if version == 1:
        compare_strategy_name = 'strategy'
        strategy_name = 'strategy_1120'
    else:
        compare_strategy_name = 'strategy_V2'
        strategy_name = 'strategy_V2_1120'
    model = 'gpt-4o-2024-11-20'

    right = {}

    for task in task_list:
        files = os.listdir(os.path.join(dir_name, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        files = files[:30]
        for file in tqdm(files, desc=task):
            with open(os.path.join(dir_name, task, file), 'r') as f:
                data = json.load(f)
            if strategy_name in data:
                for key in data[strategy_name]['final']:
                    if key not in right:
                        right[key] = []
                    right[key].append(data[strategy_name]['final'][key] == data[compare_strategy_name]['final'][key])
                continue
            try:
                data[strategy_name]= multi_cls(conv_format(data['history']), model, version, False)
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

    for key in right:
        print(f'{key}: {sum(right[key]) / len(right[key])}')

def cls_human(version: int):
    dir_name = HUMAN_DIR
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    if version == 1:
        compare_strategy_name = 'strategy'
        strategy_name = 'strategy_1120'
    else:
        compare_strategy_name = 'strategy_V2'
        strategy_name = 'strategy_V2_1120'
    model = 'gpt-4o-2024-11-20'

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
                    for key in data[strategy_name]['final']:
                        if key not in right:
                            right[key] = []
                        right[key].append(data[strategy_name]['final'][key] == data[compare_strategy_name]['final'][key])
                    continue
                try:
                    data[strategy_name]= multi_cls(conv_format(data['history']), model, version, True)
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

    for key in right:
        print(f'{key}: {sum(right[key]) / len(right[key])}')

if __name__ == '__main__':
    cls_sim(1)
    # cls_human(1)
    # cls_sim(2)
    # cls_human(2)
