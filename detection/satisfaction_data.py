import os
import random
import json
from sklearn.model_selection import train_test_split

from utils import get_profile, conv_format, HUMAN_DIR

def get_data(sample: bool = False, training: bool = True, binary: bool = False) -> tuple[list[dict], dict[int, str]] | tuple[list[dict], list[dict]]:
    random.seed(42)
    dir_name = HUMAN_DIR
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    data_list = []
    users = os.listdir(dir_name)
    users.sort()
    for user in users:
        user_data_list = []
        has_negative = False
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
                    gt = int(utt['satisfaction'])
                    user_data_list.append({
                        'task': task,
                        'user': user,
                        'file_path': os.path.join(dir_name, user, task, file),
                        'history': conv_format(data['history'][:i + 1]),
                        'profile': get_profile(data['profile']),
                        'task_context': data['task_context'],
                        'ground_truth': (0 if gt <= 3 else 1) if binary else gt
                    })
                    if binary and gt <= 3:
                        has_negative = True
        if has_negative:
            data_list.extend(user_data_list)
        else:
            print(f"User {user} has no negative samples, skipping...")
    print(f"Total data: {len(data_list)}")
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    if sample:
        test_data = test_data[:100]
    if training:
        return test_data, train_data

    examples = {}
    scores = [1, 2, 3, 4, 5] if not binary else [0, 1]
    for score in scores:
        score_list = [data for data in train_data if data['ground_truth'] == score]
        assert len(score_list) > 0, f"Score {score} not found in the data!"
        example = random.choice(score_list)['history']
        examples[score] = example
    return test_data, examples
