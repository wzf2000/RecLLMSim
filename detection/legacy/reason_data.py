import os
import random
import json
from typing import Literal, overload
from sklearn.model_selection import train_test_split

from utils import get_profile, conv_format, HUMAN_DIR

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

@overload
def get_reason_data(sample: bool = False, training: Literal[False] = False) -> tuple[list[dict], dict[int, str]]: ...

@overload
def get_reason_data(sample: bool = False, training: Literal[True] = True) -> tuple[list[dict], list[dict]]: ...

def get_reason_data(sample: bool = False, training: bool = True) -> tuple[list[dict], dict[int, str]] | tuple[list[dict], list[dict]]:
    random.seed(42)
    dir_name = HUMAN_DIR
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    data_list = []
    users = os.listdir(dir_name)
    users.sort()
    for user in users:
        for task in task_list:
            if not os.path.exists(os.path.join(dir_name, user, task)):
                continue
            files = os.listdir(os.path.join(dir_name, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(dir_name, user, task, file), 'r') as f:
                    data = json.load(f)
                if 'questionnaire' not in data:
                    continue
                for i, utt in enumerate(data['history']):
                    if utt['role'] != 'assistant':
                        continue
                    assert 'satisfaction' in utt
                    satisfaction = int(utt['satisfaction'])
                    if satisfaction > 3:
                        continue
                    assert 'reason' in utt
                    gt = utt['reason']
                    if gt == '其它':
                        continue
                    data_list.append({
                        'task': task,
                        'user': user,
                        'file_path': os.path.join(dir_name, user, task, file),
                        'history': conv_format(data['history'][:i + 1]),
                        'origin_response': data['history'][i]['content'],
                        'user_response': data['history'][i + 1]['content'] if i + 1 < len(data['history']) else '',
                        'origin_history': data['history'][:i + 1],
                        'profile': get_profile(data['profile']),
                        'task_context': data['task_context'],
                        'satisfaction': satisfaction,
                        'ground_truth': gt_map[gt]
                    })
    print(f"Total data: {len(data_list)}")
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    if sample:
        test_data = test_data[:100]
    if training:
        return test_data, train_data

    examples = {}
    scores = [0, 1, 2, 3]
    for score in scores:
        score_list = [data for data in train_data if data['ground_truth'] == score and len(data['origin_history']) >= 2]
        assert len(score_list) > 0, f"Score {score} not found in the data!"
        example = random.choice(score_list)['origin_history']
        example = example[-2:]
        example = conv_format(example)
        examples[score] = example
    return test_data, examples

@overload
def get_reason_data2(sample: bool = False, training: Literal[False] = False) -> tuple[list[dict], dict[int, str]]: ...

@overload
def get_reason_data2(sample: bool = False, training: Literal[True] = True) -> tuple[list[dict], list[dict]]: ...

def get_reason_data2(sample: bool = False, training: bool = True) -> tuple[list[dict], dict[int, str]] | tuple[list[dict], list[dict]]:
    random.seed(42)
    dir_name = HUMAN_DIR
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    user_data_list = {}
    users = os.listdir(dir_name)
    users.sort()
    for user in users:
        user_data_list[user] = []
        for task in task_list:
            if not os.path.exists(os.path.join(dir_name, user, task)):
                continue
            files = os.listdir(os.path.join(dir_name, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(dir_name, user, task, file), 'r') as f:
                    data = json.load(f)
                if 'questionnaire' not in data:
                    continue
                for i, utt in enumerate(data['history']):
                    if utt['role'] != 'assistant':
                        continue
                    assert 'satisfaction' in utt
                    satisfaction = int(utt['satisfaction'])
                    if satisfaction > 3:
                        continue
                    assert 'reason' in utt
                    gt = utt['reason']
                    if gt == '其它':
                        continue
                    user_data_list[user].append({
                        'task': task,
                        'user': user,
                        'file_path': os.path.join(dir_name, user, task, file),
                        'history': conv_format(data['history'][:i + 1]),
                        'origin_response': data['history'][i]['content'],
                        'user_response': data['history'][i + 1]['content'] if i + 1 < len(data['history']) else '',
                        'origin_history': data['history'][:i + 1],
                        'profile': get_profile(data['profile']),
                        'task_context': data['task_context'],
                        'satisfaction': satisfaction,
                        'ground_truth': gt_map[gt]
                    })
    total_len = sum([len(user_data_list[user]) for user in user_data_list])
    print(f"Total data: {total_len}")
    # sample one data for test each user
    train_data = []
    test_data = []
    for user in user_data_list:
        if len(user_data_list[user]) == 0:
            continue
        index = random.randint(0, len(user_data_list[user]) - 1)
        test_data.append(user_data_list[user][index])
        for i in range(len(user_data_list[user])):
            if i == index:
                continue
            train_data.append(user_data_list[user][i])
    if sample:
        test_data = test_data[:100]
    if training:
        return test_data, train_data

    examples = {}
    scores = [0, 1, 2, 3]
    for score in scores:
        score_list = [data for data in train_data if data['ground_truth'] == score and len(data['origin_history']) >= 2]
        assert len(score_list) > 0, f"Score {score} not found in the data!"
        example = random.choice(score_list)['origin_history']
        example = example[-2:]
        example = conv_format(example)
        examples[score] = example
    return test_data, examples
