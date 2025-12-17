import os
import json
import random
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

class Result(BaseModel):
    rewritten_texts: list[str]

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

SYSTEM_PROMPT = """你是一名专业的文本风格改写助手。
你的任务是根据用户提供的 **人类示例风格（HUMAN_STYLE_EXAMPLES）**，将 **原始语句（SOURCE_TEXTS）** 改写为自然、连贯、符合真实人类表达习惯的文本，同时保持主要含义不变。

### 你必须遵守以下规则：

* 原始输入可能包含多条语句或对话，请整体改写，使表达更自然、人类化。
* 保留核心语义：需求、时间、地点、限制条件、偏好等信息不得改变。
* 禁止生成无意义的认可或评价性句子：包括但不限于“挺好的”“这个不错”“看起来合适”“都挺符合的””这部分明确了“等类似表达。凡是对信息的认可、重复、寒暄式过渡、赞同式语气，若 HUMAN_STYLE_EXAMPLES 中未出现，则一律不得生成。模型应直接表达需求、条件或问题，不得先进行评价。
* 请删除无意义的重复、认可、寒暄、感谢等机械性内容，保持对话连贯即可。
* 需要注意模仿人类示例中对于标点、空格的使用习惯，如用户可能习惯在最后使用句号结尾，或者喜欢使用空格分隔句子等。
* 语言风格需参考 HUMAN_STYLE_EXAMPLES：包括用词、句式、语气等方面，确保改写后的文本风格与示例一致，注意不要永远保持礼貌，而是根据示例灵活调整。
* 不引入原文中没有的新信息，也不扩写多余细节。
* 多条 SOURCE_TEXTS 需要分别改写对应的输出。
* 必须按照用户指定的 JSON 输出格式，且只输出 JSON，不包含其他文字。"""

PROMPT_TEMPLATE = """下面是人类示例风格与待改写的原始文本，请根据系统指令进行改写。
你必须严格按照以下 JSON 结构输出最终结果：
```json
{{
  "rewritten_texts": [
    "string — 改写后的文本，可以一条或多条，根据内容自然拆分",
    "... more items if needed"
  ]
}}
```

**人类示例风格（HUMAN_STYLE_EXAMPLES）：**
{HUMAN_STYLE_EXAMPLES}

**需要改写的原始文本（SOURCE_TEXTS）：**
{SOURCE_TEXTS}

请输出改写后的版本。"""

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1), reraise=True)
def rewrite_texts(human_style_examples: list[str], source_texts: list[str]) -> list[str]:
    sources = [text.replace('\n', ' ').strip() for text in source_texts if text.strip()]
    prompt = PROMPT_TEMPLATE.format(
        HUMAN_STYLE_EXAMPLES="\n".join(f"- {example}" for example in human_style_examples),
        SOURCE_TEXTS="\n".join(f"- {text}" for text in sources)
    )

    response = client.chat.completions.parse(
        model="gpt-5-2025-08-07",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format=Result,
        reasoning_effort="none",
    )
    result = response.choices[0].message.parsed
    assert result is not None, "Parsed result is None"
    assert len(result.rewritten_texts) == len(source_texts), f"Number of rewritten texts {len(result.rewritten_texts)} does not match source texts {len(source_texts)}, result: {result.rewritten_texts}"
    return result.rewritten_texts

def process_file(input_file: str) -> bool:
    # 获取文件名
    file_name = os.path.basename(input_file)
    pid, tid = file_name.split('.')[0].split('_')
    pid, tid = int(pid), int(tid)
    with open("simulation_user_map.json", "r") as f:
        user_dir_list: list[str] = json.load(f)
    user_dir = user_dir_list[pid]
    # 0-4: 旅行规划, 5-8: 技能学习规划, 9-14: 礼物准备, 15-18: 菜谱规划
    task = '旅行规划' if tid <= 4 else '技能学习规划' if tid <= 8 else '礼物准备' if tid <= 14 else '菜谱规划'
    if not os.path.exists(os.path.join('..', user_dir, task)):
        # print(f"User directory {user_dir} does not have task {task}.")
        return False
    files = os.listdir(os.path.join('..', user_dir, task))
    files = [file for file in files if file.endswith('.json')]
    files.sort(key=lambda x: x.split('.')[0])
    # 拿出所有文件中的user回复
    all_user_replies = []
    for file in files:
        with open(os.path.join('..', user_dir, task, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for turn in data['history']:
            if turn['role'] == 'user':
                all_user_replies.append(turn['content'])
    # 随机选取5条作为人类示例风格
    human_style_examples = random.sample(all_user_replies, min(5, len(all_user_replies)))
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'content_rewritten' in data['history'][0]:
        # print(f"File {input_file} already processed.")
        return True
    source_texts = [turn['content'] for turn in data['history'] if turn['role'] == 'user']
    rewritten_texts = rewrite_texts(human_style_examples, source_texts)
    # 增加新字段表示改写结果
    for i in range(len(data['history'])):
        if data['history'][i]['role'] == 'user':
            data['history'][i]['content_rewritten'] = rewritten_texts.pop(0)
    # 保存回文件
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return True

def process_dir(task_dir: str):
    files = os.listdir(task_dir)
    files = [file for file in files if file.endswith('.json')]
    files.sort(key=lambda x: x.split('.')[0])
    failed = 0
    success = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_file, os.path.join(task_dir, file)): file for file in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing files in {task_dir}"):
            file = futures[future]
            try:
                flag = future.result()
                if flag:
                    success += 1
            except Exception as e:
                failed += 1
                print(f"Error processing file {os.path.join(task_dir, file)}: {e}")
    print(f"Finished processing directory {task_dir}. Failed files: {failed} / {len(files)}, Success: {success} / {len(files)}")

if __name__ == '__main__':
    for task in ['旅行规划', '技能学习规划', '礼物准备', '菜谱规划']:
        task_dir = os.path.join('../LLM_agent_user_V3', task)
        process_dir(task_dir)
