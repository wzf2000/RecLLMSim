import os
import json
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def generate(messages: list[dict[str, str]], model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        timeout=60,
    ).choices[0].message
    if response.content:
        content = response.content.strip()
        if content.startswith('<think>'):
            content = content.split('</think>')[1].strip()
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        return content
    elif response.refusal:
        print(f"Refusal: {response.refusal}")
        assert False, "Refusal"
    else:
        print("Error!")
        assert False, "Error"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def predict(messages: list[dict[str, str]], model: str) -> tuple[int, str]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        timeout=60,
    ).choices[0].message
    if response.content:
        content = response.content.strip()
        if content.startswith('<think>'):
            content = content.split('</think>')[1].strip()
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        if content.startswith('{'):
            parsed = json.loads(content)
            if hasattr(response, 'reasoning_content'):
                return parsed['classification'], response.reasoning_content + '\n\n' + response.content
            else:
                return parsed['classification'], response.content
        else:
            assert content[0] in ['0', '1', '2', '3', '4', '5'], f"Invalid prediction: {response.content}"
            if hasattr(response, 'reasoning_content'):
                return int(content[0]), response.reasoning_content + '\n\n' + response.content
            else:
                return int(content[0]), response.content
    elif response.refusal:
        print(f"Refusal: {response.refusal}")
        assert False, "Refusal"
    else:
        print("Error!")
        assert False, "Error"

def _predict_process(messages_list: list[list[dict[str, str]]], model: str, output_file: str) -> list[int]:
    total_len = len(messages_list)
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            predictions = json.load(f)
        if len(predictions) != total_len:
            assert len(predictions) < total_len, f"Prediction length mismatch: {len(predictions)} vs {total_len}"
            predictions.extend([None for _ in range(total_len - len(predictions))])
    else:
        predictions = [None for _ in range(total_len)]
    cache_dir = output_file.replace('.json', '_cache')
    os.makedirs(cache_dir, exist_ok=True)

    output_lock = Lock()

    def process_single_task(messages: list[dict[str, str]], index: int):
        if predictions[index] is not None:
            return index
        result, raw_output = predict(messages, model)
        with output_lock:
            predictions[index] = result
            with open(output_file, 'w') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=4)
            with open(os.path.join(cache_dir, f'{index}.txt'), 'w') as f:
                f.write(raw_output)
        return index

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_index = {
            executor.submit(process_single_task, messages, i): i
            for i, messages in enumerate(messages_list)
        }
        for future in tqdm(as_completed(future_to_index), total=total_len):
            try:
                _ = future.result()
            except Exception as e:
                print(f"Index {future_to_index[future]} failed with error: {e}")
    return predictions

def get_messages(input_prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a skilled conversational analyst."
        },
        {
            "role": "user",
            "content": input_prompt
        }
    ]

def predict_llm(prompts: list[str], model: str, output_file: str) -> list[int]:
    messages_list = []
    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": "You are a skilled conversational analyst."
            },
            {
                "role": "user",
                "content": prompt
            }
        ] if "r1" not in model and "reasoner" not in model else [
            {
                "role": "user",
                "content": prompt
            }
        ]
        messages_list.append(messages)
    return _predict_process(messages_list, model, output_file)

def predict_llm_in_context(prompts: list[str], histories: list[dict[str, str]], model: str, output_file: str) -> list[int]:
    messages_list = []
    for prompt, history in zip(prompts, histories):
        messages = history + [
            {
                "role": "user",
                "content": prompt
            }
        ]
        messages_list.append(messages)
    return _predict_process(messages_list, model, output_file)
