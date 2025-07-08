import os
import json

from basic_info import HUMAN_DIR, OUTPUT_DIR

def count_filed(strategy_field: str, heads: list[str], chat_model: str = None, only_all: bool = False):
    if only_all:
        task_lists = [['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']]
    else:
        task_lists = [['旅行规划', '礼物准备', '菜谱规划', '技能学习规划'], ['旅行规划'], ['菜谱规划'], ['礼物准备'], ['技能学习规划']]
    for task_list in task_lists:
        counts = {}

        def update_dir(dir_name: str):
            users = os.listdir(dir_name)
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
                        model = data.get('chat_model', 'gpt-4-turbo-preview')
                        if chat_model is not None and model != chat_model:
                            continue
                        strategy = data[strategy_field]['final']
                        for key in strategy:
                            if key not in counts:
                                counts[key] = {}
                            if strategy[key] not in counts[key]:
                                counts[key][strategy[key]] = 0
                            counts[key][strategy[key]] += 1

        update_dir(HUMAN_DIR)
        update_dir(os.path.join(HUMAN_DIR, '..', 'human_exp_V2'))

        if not os.path.exists(os.path.join(file_dir, f'count_human_{strategy_field}.txt')):
            with open(os.path.join(file_dir, f'count_human_{strategy_field}.txt'), 'w') as f:
                f.write('\t'.join(heads) + '\n')
        with open(os.path.join(file_dir, f'count_human_{strategy_field}.txt'), 'a') as f:
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
    # count_filed('strategy', ['information_request_Sequential', 'information_request_Planning'])
    # count_filed('strategy_V2', ['order_Depth', 'order_Breadth', 'order_DepthBreadth', 'order_BreadthDepth'])
    # count_filed('strategy_V3', ['question_broadness_1', 'question_broadness_2', 'question_broadness_3', 'question_broadness_4', 'question_broadness_5', 'context_dependency_1', 'context_dependency_2', 'context_dependency_3', 'context_dependency_4', 'context_dependency_5', 'feedback_NoFeedback', 'feedback_Positive', 'feedback_Negative', 'feedback_Both'])
    # count_filed('strategy_V4', ['context_dependency_1', 'context_dependency_2', 'context_dependency_3', 'context_dependency_4', 'context_dependency_5', 'explanation_Frequent', 'explanation_Rare', 'explanation_NoExplanation', 'promise_HavePromise', 'promise_NoPromise', 'feedback_NoFeedback', 'feedback_Positive', 'feedback_Negative', 'feedback_Both', 'politeness_Polite', 'politeness_Neutral', 'politeness_Impolite', 'formality_Oral', 'formality_Formal'])
    # count_filed('strategy_V5', ['utility_High', 'utility_Moderate', 'utility_Low', 'operability_High', 'operability_Moderate', 'operability_Low'])
    count_filed('strategy', ['information_request_Sequential', 'information_request_Planning'], chat_model='gpt-4-turbo-preview', only_all=True)
    count_filed('strategy_V2', ['order_Depth', 'order_Breadth', 'order_DepthBreadth', 'order_BreadthDepth'], chat_model='gpt-4-turbo-preview', only_all=True)
    count_filed('strategy_V3', ['question_broadness_1', 'question_broadness_2', 'question_broadness_3', 'question_broadness_4', 'question_broadness_5', 'context_dependency_1', 'context_dependency_2', 'context_dependency_3', 'context_dependency_4', 'context_dependency_5', 'feedback_NoFeedback', 'feedback_Positive', 'feedback_Negative', 'feedback_Both'], chat_model='gpt-4-turbo-preview', only_all=True)
    count_filed('strategy_V4', ['explanation_Frequent', 'explanation_Rare', 'explanation_NoExplanation', 'promise_HavePromise', 'promise_NoPromise', 'feedback_NoFeedback', 'feedback_Positive', 'feedback_Negative', 'feedback_Both', 'politeness_Polite', 'politeness_Neutral', 'politeness_Impolite', 'formality_Oral', 'formality_Formal'], chat_model='gpt-4-turbo-preview', only_all=True)
    count_filed('strategy_V5', ['utility_High', 'utility_Moderate', 'utility_Low', 'operability_High', 'operability_Moderate', 'operability_Low'], chat_model='gpt-4-turbo-preview', only_all=True)
