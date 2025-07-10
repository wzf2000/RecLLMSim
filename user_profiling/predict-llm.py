import os
import json
import httpx
import hashlib
import numpy as np
from tqdm import tqdm
from openai import OpenAI, LengthFinishReasonError
from pydantic import BaseModel
from argparse import ArgumentParser

from data_util import ModelType, item_translation
from evaluate_util import compute_metrics
from pipe_util import exp_sim, exp_sim2human, exp_sim2human2, exp_human, exp_human2sim, exp_human2sim2

class Answer(BaseModel):
    answer: list[int]

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

prompt = """请根据以下对话，为对话中的user选择最符合其[Profile Item]的几个描述选项（你回答的选项数不少于3个，不超过5个，并按符合度从大到小排列）：
<对话开始>
[Insert Dialogue Here]
<对话结束>

选项：
[Insert Choices Here]

请用以下JSON格式回答：
```json
{
    "answer": [2, 0, 4] // 选择的选项序号, 从0开始，数值小于选项数，长度不少于3，不超过5，并按符合度从大到小排列
}
```

你的回答是：
"""

def rank2prob(labels: np.ndarray, rank: list[int]) -> np.ndarray:
    prob = [0.0] * len(labels)
    for i, ele in enumerate(rank):
        prob[ele] = 1.0 - i * 0.01
    return np.array(prob)

cache_dir = os.path.join(os.path.dirname(__file__), 'cache')

def predict(model: str, history: str, labels: np.ndarray, item: str) -> np.ndarray:
    hashed = hashlib.md5(history.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, model, item, f'{hashed}.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            answer = np.array(data['answer'])
            cache_labels = data['labels']
            answer_label = [cache_labels[i] for i in answer]
            labels_list = labels.tolist()
            ranks = [labels_list.index(label) for label in answer_label if label in labels_list]
            return rank2prob(labels, ranks)
    else:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    label_desc = ''
    for i, label in enumerate(labels):
        label_desc += f"{i}: {label}\n"
    input_text = prompt.replace('[Insert Dialogue Here]', history).replace('[Insert Choices Here]', label_desc).replace('[Profile Item]', item_translation[item])
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled conversational analyst."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            temperature=0.0,
            response_format=Answer,
        ).choices[0].message
        if response.parsed:
            answer = response.parsed.answer
            if len(answer) < 3:
                print("Too few options!")
                return predict(model, history, labels, item)
            if len(answer) > 5:
                print("Too many options!")
                return predict(model, history, labels, item)
            with open(f'cache/{model}/{item}/{hashed}.json', 'w') as f:
                output_json = {
                    'answer': answer,
                    'history': history,
                    'labels': labels.tolist(),
                    'item': item,
                }
                json.dump(output_json, f, ensure_ascii=False, indent=4)
            return rank2prob(labels, answer)
        elif response.refusal:
            print("Refusal!")
            return predict(model, history, labels, item)
    except LengthFinishReasonError as e:
        print(f"Too many tokens: {e}")
    except Exception as e:
        print(f"Error: {e}")
    return predict(model, history, labels, item)

def work(X_train: np.ndarray, y_train: np.ndarray, X_test: list[str], y_test: np.ndarray, item: str, model_name: str, labels: np.ndarray, **kwargs) -> dict[str, float]:
    # model ID item
    y_scores = np.array([predict(model_name, history, labels, item) for history in tqdm(X_test, desc=f'Predicting {item}')])
    return compute_metrics(y_test, y_scores)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, required=True, choices=['sim', 'sim2human', 'human', 'human2sim', 'sim2human2', 'human2sim', 'human2sim2'])
    parser.add_argument('-l', '--language', type=str, default='zh', choices=['zh', 'en'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.type == 'sim':
        exp_sim(args.model, ModelType.LLM, work, args.language)
    elif args.type == 'human':
        exp_human(args.model, ModelType.LLM, work)
    elif args.type == 'sim2human':
        exp_sim2human(args.model, ModelType.LLM, work)
    elif args.type == 'human2sim':
        exp_human2sim(args.model, ModelType.LLM, work)
    elif args.type == 'sim2human2':
        exp_sim2human2(args.model, ModelType.LLM, work)
    elif args.type == 'human2sim':
        exp_human2sim(args.model, ModelType.LLM, work)
    elif args.type == 'human2sim2':
        exp_human2sim2(args.model, ModelType.LLM, work)
    else:
        raise NotImplementedError
