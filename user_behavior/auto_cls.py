import os
import json
import argparse
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
            with open(file, 'r') as f:
                data = json.load(f)
            if strategy_name not in data:
                try:
                    data[strategy_name] = multi_cls(conv_format(data['history']), model, version, False)
                except Exception:
                    print(f'Error processing {file}')
                    return
                with open(file, 'w') as fw:
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

def parse_args():
    parser = argparse.ArgumentParser(description='Classify user strategies based on model and version.')
    parser.add_argument('-t', '--task', type=str, choices=['sim', 'human'], required=True,
                        help='Task type: "sim" for simulated data, "human" for human data.')
    parser.add_argument('-v', '--version', type=int, required=True, help='Version of the classification model.')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model to use for classification.')
    parser.add_argument('-s', '--strategy_name', type=str, required=True, help='Name of the strategy to classify.')
    parser.add_argument('-c', '--compare_strategy_name', type=str, default=None, help='Name of the strategy to compare against.')
    parser.add_argument('-d', '--dir_name', type=str, default=None, help='Directory name for the data.')
    parser.add_argument('--sample', action='store_true', help='Sample a subset of data for classification.')
    args = parser.parse_args()
    if args.dir_name is None:
        args.dir_name = SIM_DIR if args.task == 'sim' else HUMAN_DIR
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'sim':
        cls_sim(args.version, args.model, args.strategy_name, args.compare_strategy_name, args.sample, args.dir_name)
    elif args.task == 'human':
        cls_human(args.version, args.model, args.strategy_name, args.compare_strategy_name, args.dir_name)
    else:
        raise ValueError("Invalid task type. Use 'sim' for simulated data or 'human' for human data.")
# Example usage:
# python auto_cls.py -t sim -v 4 -m gpt-4o-2024-08-06 -s strategy_V4 -d ../LLM_agent_user --sample
# python auto_cls.py -t human -v 4 -m gpt-4o-2024-08-06 -s strategy_V4 -d ../real_human_user
