import os
import json
from enum import Enum
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI, LengthFinishReasonError
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_fixed
from concurrent.futures import ThreadPoolExecutor, as_completed

from basic_info import SIM_DIR

api_config_file = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

prompt_template = """Now, the conversation is over. Please rate the assistant's answer based on your judgment. The specific standards are as follows:

1. **Detail**: How detailed is the response? Does it provide enough information to be actionable?
    - 0: Not detailed enough, missing key information.
    - 1: Somewhat detailed, but lacks certain specifics.
    - 2: Very detailed, all necessary information is provided.

2. **Practical Utility**: How practical is the response? Can it be realistically implemented?
    - 0: Not practical, unrealistic or overly idealistic.
    - 1: Somewhat practical, could give some suggestions but not fully actionable.
    - 2: Highly practical, can be easily implemented.

3. **Diversity**: How diverse are the options provided? Are there multiple approaches or solutions?
    - 0: Not diverse, only one approach or solution is provided.
    - 1: Somewhat diverse, a few options are given but not enough.
    - 2: Very diverse, multiple approaches or solutions are provided.

**Output Format:**

The output should be in JSON format, structured like this:
```json
{
    "detail": 0-2,
    "utility": 0-2,
    "diversity": 0-2
}
```

**Your Response:**"""


class Score(int, Enum):
    unsatisfied = 0
    ordinary = 1
    satisfied = 2


class Satsifaction(BaseModel):
    detail: Score
    utility: Score
    diversity: Score


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def label(history: list[ChatCompletionMessageParam], model: str) -> Satsifaction:
    input_text = prompt_template
    messages = history.copy()
    messages.append(ChatCompletionUserMessageParam(
        content=input_text,
        role='user',
    ))
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format=Satsifaction,
    )
    choice = response.choices[0]
    if choice.finish_reason == LengthFinishReasonError:
        print(f"Length finish reason error: {choice}")
        raise ValueError("Length finish reason error.")
    if choice.message.parsed is None:
        print(f"Response parsing failed: {choice}")
        raise ValueError("Response parsing failed.")
    ret = choice.message.parsed
    return ret


def format_history(history: list[dict[str, str]], preference: str, task_context: str) -> list[ChatCompletionMessageParam]:
    messages = [
        ChatCompletionUserMessageParam(
            content=f"Assuming I am an intelligent chat assistant, I would like you to play the role of a user who will engage in a conversation with LLM (which is me) and complete the following task requirements. (LLM refers to me):\n{task_context}\nPlease respond with \"Got it.\" to indicate your understanding.",
            role='user',
        ),
        ChatCompletionAssistantMessageParam(
            content="Got it.",
            role='assistant',
        ),
        ChatCompletionUserMessageParam(
            content=f"In the conversation, I'd like you to role-play according to the provided user profile:\n{preference}\nPlease respond with \"Got it\" to indicate understanding.",
            role='user',
        ),
        ChatCompletionAssistantMessageParam(
            content="Got it.",
            role='assistant',
        ),
        ChatCompletionUserMessageParam(
            content="""Next, please proceed as follows:

1. Please remember that if my responses don't meet your requirements, you should express it. You are not obliged to be interested in all of my answers.

2. You should make decisions that are not explicitly stated in the user profile and task requirements. Remember that you are playing the role of the user and completing tasks, not making decisions or planning on my behalf.

3. After receiving my responses, you can adjust your emotional state. For example, you can shift from expecting to being neutral or dissatisfied if you're not happy with the answers.

4. What you need is thorough planning. For each requirement, you need a specific option rather than a broad list. Also, the requirements mentioned in the task background are your priorities. You can reject and correct my suggestions if they don't meet those requirements.

5. Please avoid ending the conversation too quickly. You should achieve all the expected objectives during the conversation.

Please respond with \"Got it\" to indicate understanding.""",
            role='user',
        ),
        ChatCompletionAssistantMessageParam(
            content="Got it.",
            role='assistant',
        ),
        ChatCompletionUserMessageParam(
            content="Now, let's start the conversation. My first line is: \n\"Hello, may I ask how I can assist you?\"",
            role='user',
        ),
    ]
    history_params = [
        ChatCompletionUserMessageParam(
            content=utt['content'] + '\n[System]: If you wish to conclude this conversation, please reply with "Thank you, goodbye."',
            role='user'
        ) if utt['role'] == 'assistant' else ChatCompletionAssistantMessageParam(
            content=utt['content'],
            role='assistant'
        ) for utt in history
    ]
    history_params.append(ChatCompletionAssistantMessageParam(
        content="Thank you, goodbye.",
        role='assistant'
    ))
    messages.extend(history_params)
    return messages

def label_sim(model: str, label_name: str, sample: bool = False, dir_name: str = SIM_DIR):
    task_list = ['new travel planning', 'preparing gifts',
                 'travel planning', 'recipe planning', 'skills learning planning']

    for task in task_list:
        if not os.path.exists(os.path.join(dir_name, task)):
            continue
        files = os.listdir(os.path.join(dir_name, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        if sample:
            files = files[:30]

        def process_single(file: str):
            with open(os.path.join(dir_name, task, file), 'r') as f:
                data = json.load(f)
            if label_name not in data:
                try:
                    data[label_name] = label(format_history(
                        data['history'], data['preference'], data['task_context']), model).model_dump()
                except Exception:
                    print(f'Error processing {file}')
                    return
                with open(os.path.join(dir_name, task, file), 'w') as fw:
                    json.dump(data, fw, ensure_ascii=False, indent=4)
            return data[label_name]

        collected = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for file in files:
                futures.append(executor.submit(
                    process_single, os.path.join(dir_name, task, file)))
            for future in tqdm(as_completed(futures), total=len(files), desc=f'Processing {task}'):
                result = future.result()
                collected.append(result)
        detail = sum(item['detail'] for item in collected) / len(collected)
        utility = sum(item['utility'] for item in collected) / len(collected)
        diversity = sum(item['diversity'] for item in collected) / len(collected)
        print(f'Task: {task}, Detail: {detail:.2f}, Utility: {utility:.2f}, Diversity: {diversity:.2f}')

if __name__ == "__main__":
    label_sim('gpt-4o-2024-08-06', 'satisfaction', sample=False, dir_name=SIM_DIR)
