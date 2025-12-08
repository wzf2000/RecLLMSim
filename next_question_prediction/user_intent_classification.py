import os
import sys
import json
from tqdm import tqdm
from openai import OpenAI
from threading import Lock
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from concurrent.futures import ThreadPoolExecutor, as_completed

from nqp_data import get_nqp_data, get_nqp_data_sim, get_task_list
from utils import conv_format, SIM_DIR, SIM_DIR_V2

class IntentClassificationResult(BaseModel):
    label: str
    reason: str

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

schema = json.load(open("user_intent_schema.json"))
labels_text = "\n".join([f"{i + 1}. {l['label']}: {l['definition']}" for i, l in enumerate(schema["labels"])])

SYSTEM_PROMPT = f"""你是一名对话分析助手，任务是根据用户的发言内容识别其主要意图或行为特点。

你将收到一条用户发言文本（可包含上下文），请根据以下分类体系选择最合适的标签。
每条发言只选择一个最主要的类别。

输出格式为 JSON：
{{"label": "类别名称", "reason": "简要说明选择理由"}}

分类体系：
{labels_text}
"""

USER_PROMPT = """以下是用户与AI的对话历史以及本轮用户发言：

历史对话：
{history_text}

用户发言：
{user_text}

对话任务背景：
{task_description}

请判断该用户此轮发言最符合以下哪个意图类别，并给出简短解释。"""

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def classify_turn(data: dict) -> IntentClassificationResult:
    input_text = USER_PROMPT.format(
        history_text=conv_format(data['origin_history'][-4:]),
        user_text=data['user_question'],
        task_description=data['task_context'],
    )
    response = client.beta.chat.completions.parse(
        model="gpt-5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ],
        temperature=0.0,
        response_format=IntentClassificationResult,
        reasoning_effort="none",
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse response")
    return parsed

def main(human: bool = True, version: int = 1):
    file_lock_dict = {}
    dict_lock = Lock()
    if human:
        train_data, test_data = get_nqp_data(sample=False, task_list=get_task_list(human=True, task_name='all'))
        data = train_data + test_data
    else:
        data = get_nqp_data_sim(task_list=get_task_list(human=False, task_name='all'), sim_dir=SIM_DIR_V2 if version == 2 else SIM_DIR)

    def process_single(i: int, item: dict):
        with dict_lock:
            file_lock = file_lock_dict.setdefault(item['file_path'], Lock())
        with file_lock:
            with open(item['file_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data['history'][item['turn']].get('intent_label', None) is not None:
                return
            result = classify_turn(item)
            data['history'][item['turn']]['intent_label'] = result.label
            data['history'][item['turn']]['intent_reason'] = result.reason
            with open(item['file_path'], 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i, item in enumerate(data):
            futures.append(executor.submit(process_single, i, item))
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0 and 'sim' in args:
        if len(args) > 1 and 'v2' in args:
            main(human=False, version=2)
        else:
            main(human=False, version=1)
    else:
        main()
