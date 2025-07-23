import os
import json

from basic_info import SIM_DIR, OUTPUT_DIR

def count_filed(strategy_field: str, heads: list[str], dir_name: str):
    task_lists = [['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning'], ['new travel planning', 'travel planning'], ['recipe planning'], ['preparing gifts'], ['skills learning planning']]
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
                if 'final' in data[strategy_field]:
                    strategy = data[strategy_field]['final']
                else:
                    strategy = data[strategy_field]
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

if __name__ == '__main__':
    file_dir = OUTPUT_DIR
    os.makedirs(file_dir, exist_ok=True)
    # example usage below:
    # count_filed('strategy_V1_update', ['problem_solving_AllInOne', 'problem_solving_StepByStep', 'order_Depth', 'order_Breadth', 'order_DepthBreadth', 'order_BreadthDepth'], SIM_DIR)
    # count_filed('strategy_V3', ['question_broadness_1', 'question_broadness_2', 'question_broadness_3', 'question_broadness_4', 'question_broadness_5', 'context_dependency_1', 'context_dependency_2', 'context_dependency_3', 'context_dependency_4', 'context_dependency_5'], SIM_DIR)
    # count_filed('satisfaction', ['detail_0', 'detail_1', 'detail_2', 'utility_0', 'utility_1', 'utility_2', 'diversity_0', 'diversity_1', 'diversity_2'], SIM_DIR)
    # count_filed('strategy_V4', ['promise_HavePromise', 'promise_NoPromise', 'feedback_NoFeedback', 'feedback_Positive', 'feedback_Negative', 'feedback_Both', 'politeness_Polite', 'politeness_Neutral', 'politeness_Impolite', 'formality_Oral', 'formality_Formal', 'formality_Mixed'], SIM_DIR)
    # count_filed('strategy_V5', ['utility_High', 'utility_Moderate', 'utility_Low', 'operability_High', 'operability_Moderate', 'operability_Low'], SIM_DIR)
