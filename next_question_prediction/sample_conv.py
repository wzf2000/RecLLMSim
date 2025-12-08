import os
import random
import json
from pydantic import BaseModel
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from utils import get_profile, conv_format, HUMAN_DIR, HUMAN_DIR_V2

class Label(BaseModel):
    label: str
    definition: str

class LabelsResponse(BaseModel):
    labels: list[Label]

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

prompt_template = """你将看到若干用户与AI的对话片段。请根据他们的行为特点，将这些用户发言分成若干类型，
每一类代表一种典型的用户需求或行为（如“要求更具体”、“提出修改”、“质疑结果”等），
并为每一类起一个简洁的标签和一句定义。

输出格式为JSON，内容形同如下：
{
    "labels": [
        {"label": "要求更具体", "definition": "用户希望AI回答更详细或更具体"},
        {"label": "提出额外要求", "definition": "用户在已有结果基础上提出新的需求"},
        ...
    ]
}

请控制总的需求/行为类别在5到10个之间，确保每个类别都有足够的代表性。

以下是一些对话片段：
"""

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

def get_task_list(human: bool, task_name: str) -> list[str]:
    if human:
        return {
            'travel': ['旅行规划'],
            'gift': ['礼物准备'],
            'recipe': ['菜谱规划'],
            'skill': ['技能学习规划'],
            'all': ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
        }[task_name]
    else:
        return {
            'travel': ['new travel planning', 'travel planning'],
            'gift': ['preparing gifts'],
            'recipe': ['recipe planning'],
            'skill': ['skills learning planning'],
            'all': ['new travel planning', 'preparing gifts', 'recipe planning', 'skills learning planning', 'travel planning']
        }[task_name]

def get_conv_data(samples: int, task_list: list[str] | None = None) -> list[dict]:
    random.seed(42)
    data_list: list[dict] = []
    if task_list is None:
        task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']

    def get_data_from_dir(dir_name: str):
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
                    data_list.append({
                        'task': task,
                        'context': conv_format(data['history']),
                        'profile': get_profile(data['profile']),
                        'task_context': data['task_context'],
                    })

    get_data_from_dir(HUMAN_DIR)
    get_data_from_dir(HUMAN_DIR_V2)
    print(f"Total data: {len(data_list)}")
    if samples < len(data_list):
        data_list = random.sample(data_list, samples)
    return data_list

def get_conv_format(sample_convs: list[dict[str, str]]) -> str:
    convs_str = ""
    for i, conv in enumerate(sample_convs):
        convs_str += f"[对话片段 {i+1}:]\n"
        convs_str += f"[用户信息]: {conv['profile']}\n"
        convs_str += f"[任务背景]: {conv['task_context']}\n"
        convs_str += "[对话内容]:\n[Conversation]\n"
        convs_str += conv['context'] + "\n[/Conversation]\n\n"
    return convs_str

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def ask(sample_convs: list[dict[str, str]], model: str) -> LabelsResponse:
    input_text = prompt_template + get_conv_format(sample_convs)
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a skilled conversational analyst."},
            {"role": "user", "content": input_text}
        ],
        temperature=0.0,
        response_format=LabelsResponse,
    ).choices[0].message.parsed
    if response is None or not isinstance(response, LabelsResponse):
        raise ValueError("Invalid response format")
    return response

if __name__ == "__main__":
    groups = 5
    sample_per_group = 20
    total_samples = groups * sample_per_group
    conv_sample_list = get_conv_data(samples=total_samples, task_list=get_task_list(human=True, task_name='all'))
    labels = []
    with open('./labels.json', 'w') as f:
        for i in range(groups):
            sample_convs = conv_sample_list[i * sample_per_group: (i + 1) * sample_per_group]
            response = ask(sample_convs, model="gpt-5-2025-08-07")
            print(f"Group {i + 1} Labels:")
            for label in response.labels:
                print(f"- {label.label}: {label.definition}")
            print("\n")
            labels.append(response.model_dump())
        json.dump(labels, f, ensure_ascii=False, indent=2)
