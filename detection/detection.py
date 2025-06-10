import os
import json
import random

from ml import evaluate_ml
from lm import evaluate_lm
from llm import predict_llm
from utils import conv_format
from evaluation import evaluate

HUMAN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real_human_user')

prompt = """请根据以下用户和 LLM 的对话判断用户对于 LLM 回答的 Practical Utility 打分，以下为打分原则：
- **不可用 (0)：**计划不切实际（例如，过于紧凑的旅行计划或进度不合理的快速学习计划）。
- **有点可用 (1)：**建议部分适用（例如，提供灵感但难以实施的礼物创意）。
- **高度可用 (2)：**计划实用且可操作（例如，具有可实现里程碑的 Python 学习计划）。

以下是不可用（0）的对话示例：
{zero_example}

以下是有点可用（1）的对话示例：
{one_example}

以下是高度可用（2）的对话示例：
{two_example}

以下为用户进行对话的相关任务背景信息：
{task_context}

以下为你需要判断的对话：
{context}

请根据上述对话与原则，判断用户对于 LLM 回答的 Practical Utility 打分（0/1/2），注意不要输出除了 0/1/2 之外的其他字符："""

compare_prompt = """请根据以下两段用户和 LLM 的对话判断哪一段对话中用户对于 LLM 回答的 Practical Utility(总体来说，对方给出的回答对现实中真实遇到类似的问题有多少参考和使用价值)打分更高：

以下为第一段对话中用户的相关任务背景信息：
{task_context1}

以下为第一段对话：
{context1}

以下为第二段对话中用户的相关任务背景信息：
{task_context2}

以下为第二段对话：
{context2}

请根据上述两段对话与原则，判断用户对哪一段对话的 Practical Utility 打分更高（1/2），注意不要输出除了 1/2 之外的其他字符："""

def get_compare_data():
    random.seed(42)
    dir_name = HUMAN_DIR
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    data_dict = {}
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
                questionnaire = data['questionnaire']
                for item in questionnaire:
                    if item['summary'] == 'Availability':
                        gt = {
                            "很有用": 2,
                            "有一定参考": 1,
                            "不可用": 0,
                        }[item['option']]
                if user not in data_dict:
                    data_dict[user] = {}
                if task not in data_dict[user]:
                    data_dict[user][task] = []
                data_dict[user][task].append({
                    'task': task,
                    'user': user,
                    'file_path': os.path.join(dir_name, user, task, file),
                    'history': conv_format(data['history']),
                    'task_context': data['task_context'],
                    'ground_truth': gt
                })
    data_list = []
    for user in data_dict:
        for task in data_dict[user]:
            num = len(data_dict[user][task])
            if num < 2:
                continue
            for first_task in range(num - 1):
                for second_task in range(first_task + 1, num):
                    if data_dict[user][task][first_task]['ground_truth'] == data_dict[user][task][second_task]['ground_truth']:
                        continue
                    else:
                        data_list.append({
                            'task': task,
                            'user': user,
                            'history1': data_dict[user][task][first_task]['history'],
                            'history2': data_dict[user][task][second_task]['history'],
                            'task_context1': data_dict[user][task][first_task]['task_context'],
                            'task_context2': data_dict[user][task][second_task]['task_context'],
                            'ground_truth': 2 if data_dict[user][task][first_task]['ground_truth'] < data_dict[user][task][second_task]['ground_truth'] else 1
                        })
    return data_list

def get_data(sample: bool = False, data_only: bool = True) -> tuple[list[dict], str, str, str] | list[dict]:
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
                questionnaire = data['questionnaire']
                for item in questionnaire:
                    if item['summary'] == 'Availability':
                        gt = {
                            "很有用": 2,
                            "有一定参考": 1,
                            "不可用": 0,
                        }[item['option']]
                data_list.append({
                    'task': task,
                    'user': user,
                    'file_path': os.path.join(dir_name, user, task, file),
                    'history': conv_format(data['history']),
                    'task_context': data['task_context'],
                    'ground_truth': gt
                })
    if sample:
        random.shuffle(data_list)
        shot_list = data_list[30:]
        data_list = data_list[:30]
    else:
        shot_list = data_list
    if data_only:
        return data_list, shot_list

    zero_list = [data for data in shot_list if data['ground_truth'] == 2]
    one_list = [data for data in shot_list if data['ground_truth'] == 1]
    two_list = [data for data in shot_list if data['ground_truth'] == 0]
    assert len(zero_list) > 0 and len(one_list) > 0 and len(two_list) > 0
    zero_example = random.choice(zero_list)['history']
    one_example = random.choice(one_list)['history']
    two_example = random.choice(two_list)['history']
    return data_list, zero_example, one_example, two_example

def work_llm(model: str, version: int, sample: bool = False) -> tuple[float, float]:
    test_data, zero_example, one_example, two_example = get_data(sample, data_only=False)
    ground_truth = [data['ground_truth'] for data in test_data]
    output_file = f'results/human_predictions_v{version}{"_sampled" if sample else ""}.json'
    prompts = [prompt.format(context=data['history'], task_context=data['task_context'], zero_example=zero_example, one_example=one_example, two_example=two_example) for data in test_data]
    predictions = predict_llm(prompts, model, output_file=output_file)
    return evaluate(predictions, ground_truth)

def compare_work_llm(model: str, version: int) -> tuple[float, float]:
    data_list = get_compare_data()
    print(f"Total number of data: {len(data_list)}")
    predictions = []
    ground_truth = []
    for data in data_list:
        ground_truth.append(data['ground_truth'])
    output_file = f'results/human_compare_predictions_v{version}.json'
    prompts = [compare_prompt.format(context1=data['history1'], context2=data['history2'], task_context1=data['task_context1'], task_context2=data['task_context2']) for data in data_list]
    predictions = predict_llm(prompts, model, output_file=output_file)
    return evaluate(predictions, ground_truth, True)

def work_ml(vectorizer: str = 'tfidf', model_name: str = 'RF') -> tuple[float, float]:
    test_data, train_data = get_data(sample=True, data_only=True)
    return evaluate_ml(train_data, test_data, vectorizer=vectorizer, model_name=model_name)

def work_lm(model_name: str, regression: bool = False) -> tuple[float, float]:
    test_data, train_data = get_data(sample=True, data_only=True)
    results = evaluate_lm(model_name, train_data, test_data, num_labels=3, regression=regression)
    return results['eval_mse'], results['eval_rmse']

if __name__ == '__main__':
    # work_ml('count', 'RF')
    # work_ml('tfidf', 'RF')
    # work_ml('count', 'XGB')
    # work_ml('tfidf', 'XGB')
    # work_ml('count', 'RFReg')
    # work_ml('tfidf', 'RFReg')
    # work_ml('count', 'XGBReg')
    # work_ml('tfidf', 'XGBReg')
    # work_lm('bert-base-chinese', regression=False)
    # work_lm('bert-base-chinese', regression=True)
    # work_llm('gpt-4o-2024-08-06', 1, sample=False)
    # work_llm('gpt-4o-2024-08-06', 2, sample=False)
    # work_llm('gpt-4o-2024-08-06', 3, sample=True)
    # work_llm('gpt-4o-2024-08-06', 4, sample=True)
    # work_llm('gpt-4-turbo-preview', 5, sample=True)
    # work_llm('claude-3-5-sonnet-20241022', 6, sample=True)
    # work_llm('deepseek-r1', 7, sample=True)
    # get_ground_truth()

    compare_work_llm('gpt-4o-2024-08-06', 1)
