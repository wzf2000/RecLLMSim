import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from strategy_cls import multi_cls, conv_format
from basic_info import SIM_DIR, HUMAN_DIR

def cls_sim(version: int, model: str, strategy_name: str, compare_strategy_name: str | None, sample: bool = False, dir_name: str = SIM_DIR):
    task_list = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']

    if compare_strategy_name is not None:
        right = {}

    for task in task_list:
        if not os.path.exists(os.path.join(dir_name, task)):
            continue
        files = os.listdir(os.path.join(dir_name, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        if sample:
            files = files[:30]

        def compare_count(data: dict, strategy_name: str, compare_strategy_name: str):
            for key in data[strategy_name]['final']:
                if key not in right:
                    right[key] = []
                right[key].append(data[strategy_name]['final'][key] == data[compare_strategy_name]['final'][key])

        def process_single(file: str):
            with open(os.path.join(dir_name, task, file), 'r') as f:
                data = json.load(f)
            if strategy_name not in data:
                try:
                    data[strategy_name] = multi_cls(conv_format(data['history']), model, version, False)
                except Exception:
                    print(f'Error processing {file}')
                    return
                with open(os.path.join(dir_name, task, file), 'w') as fw:
                    json.dump(data, fw, ensure_ascii=False, indent=4)
            if compare_strategy_name is not None:
                compare_count(data, strategy_name, compare_strategy_name)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for file in files:
                futures.append(executor.submit(process_single, os.path.join(dir_name, task, file)))
            for future in tqdm(as_completed(futures), total=len(files)):
                future.result()

    if compare_strategy_name is not None:
        for key in right:
            print(f'{key}: {sum(right[key]) / len(right[key])}')

def cls_human(version: int, model: str, strategy_name: str, compare_strategy_name: str | None, dir_name: str = HUMAN_DIR):
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']

    if compare_strategy_name is not None:
        right = {}

    for user in tqdm(os.listdir(dir_name)):
        for task in task_list:
            if not os.path.exists(os.path.join(dir_name, user, task)):
                continue
            files = os.listdir(os.path.join(dir_name, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])

            def compare_count(data: dict, strategy_name: str, compare_strategy_name: str):
                for key in data[strategy_name]['final']:
                    if key not in right:
                        right[key] = []
                    right[key].append(data[strategy_name]['final'][key] == data[compare_strategy_name]['final'][key])

            def process_single(file: str):
                with open(os.path.join(file), 'r') as f:
                    data = json.load(f)
                if strategy_name not in data:
                    try:
                        data[strategy_name] = multi_cls(conv_format(data['history']), model, version, True)
                    except Exception:
                        print(f'Error processing {file}')
                        return
                    with open(os.path.join(file), 'w') as fw:
                        json.dump(data, fw, ensure_ascii=False, indent=4)
                if compare_strategy_name is not None:
                    compare_count(data, strategy_name, compare_strategy_name)

            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for file in files:
                    futures.append(executor.submit(process_single, os.path.join(dir_name, user, task, file)))
                for future in futures:
                    future.result()

    if compare_strategy_name is not None:
        for key in right:
            print(f'{key}: {sum(right[key]) / len(right[key])}')

if __name__ == '__main__':
    # cls_sim(1, 'gpt-4o-2024-11-20', 'strategy_1120', 'strategy', True)
    # cls_sim(2, 'gpt-4o-2024-11-20', 'strategy_V2_1120', 'strategy_V2', True)
    # cls_human(1, 'gpt-4o-2024-11-20', 'strategy_1120', 'strategy')
    # cls_human(2, 'gpt-4o-2024-11-20', 'strategy_V2_1120', 'strategy_V2')
    # cls_sim(3, 'gpt-4o-2024-08-06', 'strategy_V3', None)
    # cls_sim(3, 'gpt-4o-2024-08-06', 'strategy_V3', None, False, os.path.join(SIM_DIR, '..', 'addition_simulated', 'claude'))
    # cls_sim(3, 'gpt-4o-2024-08-06', 'strategy_V3', None, False, os.path.join(SIM_DIR, '..', 'addition_simulated', 'gemini'))
    # cls_human(3, 'gpt-4o-2024-08-06', 'strategy_V3', None, os.path.join(HUMAN_DIR, '..', 'human_exp_V2'))
    # cls_human(1, 'gpt-4o-2024-08-06', 'strategy', None, os.path.join(HUMAN_DIR, '..', 'human_exp_V2'))
    # cls_human(2, 'gpt-4o-2024-08-06', 'strategy_V2', None, os.path.join(HUMAN_DIR, '..', 'human_exp_V2'))
    # cls_human(4, 'gpt-4o-2024-08-06', 'strategy_V4', None, HUMAN_DIR)
    # cls_human(4, 'gpt-4o-2024-08-06', 'strategy_V4', None, os.path.join(HUMAN_DIR, '..', 'human_exp_V2'))
    cls_sim(4, 'gpt-4o-2024-08-06', 'strategy_V4', None, False, SIM_DIR)
