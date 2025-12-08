import os
import json
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from concurrent.futures import ThreadPoolExecutor, as_completed


class IntentPredictionResult(BaseModel):
    label: str

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

schema = json.load(open("user_intent_schema.json"))
labels_text = "\n".join([f"{i+1}. {l['label']}: {l['definition']}" for i, l in enumerate(schema["labels"])])

SYSTEM_PROMPT = f"""你是一名对话分析助手，任务是根据用户与AI的对话历史预测其下一轮发言的主要意图或行为特点。

你将收到用户与AI的完整对话历史以及对话发生的任务背景，请根据以下分类体系选择最合适的标签。
每段对话请只预测一个最可能的类别。

输出格式为 JSON：
{{"label": "类别名称"}}

请注意不要输出```json```等多余字符，确保输出仅为有效的 JSON 格式。如：{{"label": "请求具体/可执行方案"}}

分类体系：
{labels_text}
"""

USER_PROMPT = """以下是用户与AI的对话历史以及本轮用户发言

历史对话：
{history_text}

对话任务背景：
{task_description}

用户画像：
{user_profile}

请判断该用户下轮发言最可能符合以下哪个意图类别。
"""

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def classify_turn(model: str, data: dict) -> IntentPredictionResult:
    input_text = USER_PROMPT.format(
        history_text=data['history'],
        task_description=data['task_context'],
        user_profile=data['profile'],
    )
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ],
        temperature=0.0,
        response_format=IntentPredictionResult,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse response")
    return parsed

def predict_intents(model: str, data_list: list[dict], max_workers: int = 5) -> list[str]:
    intents = [None] * len(data_list)
    output_dir = os.path.join('results', 'llm', f'{model}')
    os.makedirs(output_dir, exist_ok=True)

    def process_single(i: int, item: dict):
        output_file = os.path.join(output_dir, f'results_{item["id"]}.txt')
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                intent = f.read().strip()
            intents[i] = intent
            return
        try:
            result = classify_turn(model, item)
            intents[i] = result.label
            with open(output_file, 'w') as f:
                f.write(result.label)
        except Exception as exc:
            print(f"Data {item} generated an exception: {exc}")
            intents[i] = "Unknown"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, i, item): i for i, item in enumerate(data_list)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying intents"):
            future.result()  # to raise exceptions if any
    return intents
