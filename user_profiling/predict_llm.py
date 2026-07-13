import os
import json
import hashlib
import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from argparse import ArgumentParser

from data_util import ModelType, item_translation
from evaluate_util import compute_metrics
from pipe_util import exp_sim, exp_sim2human, exp_sim2human2, exp_human, exp_human2sim, exp_human2sim2

class Answer(BaseModel):
    answer: list[int]

DEFAULT_API_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')
client: OpenAI | None = None

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

def rank2prob(labels: list[str], rank: list[int]) -> np.ndarray:
    prob = [0.0] * len(labels)
    for i, ele in enumerate(rank):
        prob[ele] = 1.0 - i * 0.01
    return np.array(prob)

cache_dir = os.path.join(os.path.dirname(__file__), 'cache')

def configure_client(config_file: str | None = None) -> None:
    global client
    config = {}
    if config_file and os.path.exists(config_file):
        with open(config_file) as file:
            config = json.load(file)
    api_key = os.environ.get('OPENAI_API_KEY', config.get('api_key'))
    base_url = os.environ.get('OPENAI_BASE_URL', config.get('base_url'))
    if not api_key:
        raise ValueError('Set OPENAI_API_KEY or provide api_key in --config')
    client_kwargs = {'api_key': api_key}
    if base_url:
        client_kwargs['base_url'] = base_url
    client = OpenAI(**client_kwargs)

def valid_answer(answer: list[int], num_labels: int) -> bool:
    return 3 <= len(answer) <= min(5, num_labels) and len(set(answer)) == len(answer) and all(0 <= index < num_labels for index in answer)

def predict(model: str, history: str, labels: np.ndarray, item: str, max_retries: int = 3) -> np.ndarray:
    cache_key = json.dumps({'history': history, 'labels': labels.tolist(), 'prompt_version': 2}, ensure_ascii=False, sort_keys=True)
    hashed = hashlib.sha256(cache_key.encode()).hexdigest()
    safe_model = model.replace('/', '__')
    cache_file = os.path.join(cache_dir, safe_model, item, f'{hashed}.json')
    if os.path.exists(cache_file):
        with open(cache_file) as file:
            data = json.load(file)
        answer = data.get('answer', [])
        cache_labels = data.get('labels', [])
        if valid_answer(answer, len(cache_labels)):
            answer_labels = [cache_labels[index] for index in answer]
            labels_list = labels.tolist()
            ranks = [labels_list.index(label) for label in answer_labels if label in labels_list]
            if valid_answer(ranks, len(labels)):
                return rank2prob(labels, ranks)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    label_desc = ''.join(f'{i}: {label}\n' for i, label in enumerate(labels))
    input_text = prompt.replace('[Insert Dialogue Here]', history).replace('[Insert Choices Here]', label_desc).replace('[Profile Item]', item_translation[item])
    if client is None:
        raise RuntimeError('LLM client is not configured')
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {'role': 'system', 'content': 'You are a skilled conversational analyst.'},
                    {'role': 'user', 'content': input_text},
                ],
                temperature=0.0,
                response_format=Answer,
            ).choices[0].message
            if response.refusal:
                raise RuntimeError(f'Model refusal: {response.refusal}')
            answer = response.parsed.answer if response.parsed else []
            if not valid_answer(answer, len(labels)):
                raise ValueError(f'Invalid ranked answer: {answer}')
            with open(cache_file, 'w') as file:
                json.dump({'answer': answer, 'labels': labels.tolist(), 'item': item}, file, ensure_ascii=False, indent=2)
            return rank2prob(labels, answer)
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'LLM profiling failed after {max_retries + 1} attempts') from last_error

def work(X_train: list[str], y_train: np.ndarray, X_test: list[str], y_test: np.ndarray, item: str, model_name: str, labels: np.ndarray, max_retries: int = 3, **kwargs) -> dict[str, float]:
    y_scores = np.array([predict(model_name, history, labels, item, max_retries) for history in tqdm(X_test, desc=f'Predicting {item}')])
    return compute_metrics(y_test, y_scores)

def list_str(value: str) -> list[str]:
    return value.split(',')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, required=True, choices=['sim', 'sim2human', 'human', 'human2sim', 'sim2human2', 'human2sim', 'human2sim2'])
    parser.add_argument('-l', '--language', type=str, default='zh', choices=['zh', 'en'])
    parser.add_argument('-d', '--data_version', type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default=DEFAULT_API_CONFIG)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--only_all', action='store_true')
    parser.add_argument('--items', type=list_str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    configure_client(args.config)
    if args.type == 'sim':
        exp_sim(args.model, ModelType.LLM, work, args.language)
    elif args.type == 'human':
        exp_human(args.model, ModelType.LLM, work, data_version=args.data_version, seed=args.seed, max_retries=args.max_retries, only_all=args.only_all, items=args.items)
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
