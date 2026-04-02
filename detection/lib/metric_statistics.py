import os
import json
import numpy as np
from loguru import logger

from .utils import HUMAN_DIR

def get_satisfaction_data(dir_name: str = HUMAN_DIR) -> list[dict[str, int | str | list[dict[str, str]]]]:
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
                history = [{
                    'role': utt['role'],
                    'content': utt['content']
                } for utt in data['history']]
                sample = {
                    'task': task,
                    'user': user,
                    'turns': len(data['history']),
                    'file_path': os.path.join(dir_name, user, task, file),
                    'history': history,
                    'profile': data['profile'],
                    'task_context': data['task_context'],
                    'chat_model': data['chat_model'],
                }
                satisfaction_scores = []
                reasons = []
                for i, utt in enumerate(data['history']):
                    if utt['role'] != 'assistant':
                        continue
                    assert 'satisfaction' in utt
                    gt = int(utt['satisfaction'])
                    satisfaction_scores.append(gt)
                    if gt <= 3:
                        assert 'reason' in utt
                        reasons.append(utt['reason'])
                    else:
                        reasons.append('满意')
                sample['satisfaction_scores'] = satisfaction_scores
                sample['dissatisfaction_reasons'] = reasons
                sample['assistant_turns'] = len(satisfaction_scores)
                data_list.append(sample)
    logger.info(f"Total data: {len(data_list)}")
    return data_list

def average_satisfaction_score(data_list: list[dict]) -> float:
    total_turns = sum(sample['assistant_turns'] for sample in data_list)
    total_satisfaction_scores = sum(sum(sample['satisfaction_scores']) for sample in data_list)
    avg_satisfaction_score = total_satisfaction_scores / total_turns
    return avg_satisfaction_score

def satisfaction_stability_score(data_list: list[dict]) -> float:
    satisfaction_stable = [all(score >= 4 for score in sample['satisfaction_scores']) for sample in data_list]
    stability_score = np.mean(satisfaction_stable)
    return stability_score

def recovery_capability_score(data_list: list[dict]) -> float:
    score_after_dissatisfaction = [[sample['satisfaction_scores'][i + 1] for i in range(len(sample['satisfaction_scores']) - 1) if sample['satisfaction_scores'][i] <= 3] for sample in data_list]
    recovery = [score >= 4 for scores in score_after_dissatisfaction for score in scores]
    recovery_score = np.mean(recovery)
    return recovery_score

def main():
    data_list = get_satisfaction_data()
    samples_group_by_model = {}
    for sample in data_list:
        model = sample['chat_model']
        if model not in samples_group_by_model:
            samples_group_by_model[model] = []
        samples_group_by_model[model].append(sample)
    for model, samples in samples_group_by_model.items():
        avg_satisfaction_score = average_satisfaction_score(samples)
        stability_score = satisfaction_stability_score(samples)
        recovery_score = recovery_capability_score(samples)
        print(f"{model}\t{avg_satisfaction_score:.4f}\t{stability_score:.4f}\t{recovery_score:.4f}")

if __name__ == '__main__':
    main()
