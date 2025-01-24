import os
import json

from basic_info import SIM_DIR, OUTPUT_DIR

dir_name = SIM_DIR
# dir_name = os.path.join(SIM_DIR, '..', 'addition_simulated', 'gemini')
file_dir = OUTPUT_DIR
strategy_field = 'strategy_V3'
heads = ['question_broadness_1', 'question_broadness_2', 'question_broadness_3', 'question_broadness_4', 'question_broadness_5', 'context_dependency_1', 'context_dependency_2', 'context_dependency_3', 'context_dependency_4', 'context_dependency_5', 'feedback_NoFeedback', 'feedback_Positive', 'feedback_Negative', 'feedback_Both']
# task_lists = [['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning'], ['new travel planning', 'travel planning'], ['recipe planning'], ['preparing gifts'], ['skills learning planning']]
task_lists = [['travel planning']]
for task_list in task_lists:
    counts = {}
    for task in task_list:
        files = os.listdir(os.path.join(dir_name, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(dir_name, task, file), 'r') as f:
                data = json.load(f)
            if strategy_field not in data:
                continue
            strategy = data[strategy_field]['final']
            for key in strategy:
                if key not in counts:
                    counts[key] = {}
                if strategy[key] not in counts[key]:
                    counts[key][strategy[key]] = 0
                counts[key][strategy[key]] += 1

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(os.path.join(file_dir, f'count_sim_{strategy_field}.txt')):
        with open(os.path.join(file_dir, f'count_sim_{strategy_field}.txt'), 'w') as f:
            f.write('\t'.join(heads) + '\n')
    with open(os.path.join(file_dir, f'count_sim_{strategy_field}.txt'), 'a') as f:
        column = []
        for col in heads:
            split = col.split('_')
            value = split[-1]
            key = '_'.join(split[:-1])
            value = int(value) if value.isdigit() else value
            column.append(str(counts.get(key, {}).get(value, 0)))
        f.write('\t'.join(column) + '\n')
