import os
import json
import tiktoken

from utils import SIM_DIR, HUMAN_DIR

tokenizer = tiktoken.encoding_for_model('gpt-4-turbo-preview')

task_translation = {
    '旅行规划': 'travel planning',
    '礼物准备': 'preparing gifts',
    '菜谱规划': 'recipe planning',
    '技能学习规划': 'skills learning planning'
}

task_translation_reverse = {
    'travel planning': '旅行规划',
    'preparing gifts': '礼物准备',
    'recipe planning': '菜谱规划',
    'skills learning planning': '技能学习规划'
}

item_translation = {
    'Personality': '性格',
    'Daily Interests and Hobbies': '日常兴趣爱好',
    'Travel Habits': '旅行习惯',
    'Dining Preferences': '饮食偏好',
    'Spending Habits': '消费习惯',
    'Other Aspects': '其他方面'
}

def get_sim_data(task: str | None = None, language: str = 'en') -> list[list[dict[str, str]]]:
    if language == 'zh' and task is not None and task in task_translation:
        task = task_translation[task]
    if task is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    elif task == 'travel planning':
        tasks = ['new travel planning', 'travel planning']
    else:
        tasks = [task]
    convs: list[list[dict[str, str]]] = []
    for task in tasks:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(SIM_DIR, task, file), 'r') as f:
                data = json.load(f)
            if language == 'en':
                history = [{
                    'role': utt['role'],
                    'content': utt['content']
                } for utt in data['history']]
            else:
                history = [{
                    'role': utt['role'],
                    'content': utt['content_zh']
                } for utt in data['history']]
            convs.append(history)
    return convs

def get_human_data(task: str | None = None) -> list[list[dict[str, str]]]:
    if task is not None and task in task_translation_reverse:
        task = task_translation_reverse[task]
    if task is None:
        tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    else:
        tasks = [task]
    convs: list[list[dict[str, str]]] = []
    users = os.listdir(HUMAN_DIR)
    for user in users:
        for task in tasks:
            if not os.path.exists(os.path.join(HUMAN_DIR, user, task)):
                continue
            files = os.listdir(os.path.join(HUMAN_DIR, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(HUMAN_DIR, user, task, file), 'r') as f:
                    data = json.load(f)
                history = [{
                    'role': utt['role'],
                    'content': utt['content']
                } for utt in data['history']]
                convs.append(history)
    return convs

def turns(conv: list[dict[str, str]]) -> int:
    return len(conv) // 2

def tokens(conv: list[dict[str, str]] | dict[str, str], role: str | None = None) -> int:
    if isinstance(conv, list):
        return sum(tokens(utt, role) for utt in conv)
    if role is not None and conv['role'] != role:
        return 0
    return len(tokenizer.encode(conv['content']))

def average(x: list[int]) -> float:
    return sum(x) / len(x)

def statistics(convs: list[list[dict[str, str]]]) -> dict[str, float | int]:
    return {
        'convs': len(convs),
        'turns': average([turns(conv) for conv in convs]),
        'avg_user_tokens': average([tokens(conv, 'user') for conv in convs]),
        'avg_assistant_tokens': average([tokens(conv, 'assistant') for conv in convs]),
        'avg_user_tokens_turn': average([tokens(utt) for conv in convs for utt in conv if utt['role'] == 'user']),
        'avg_assistant_tokens_turn': average([tokens(utt) for conv in convs for utt in conv if utt['role'] == 'assistant'])
    }

if __name__ == '__main__':
    for task in [None, 'travel planning', 'recipe planning', 'preparing gifts', 'skills learning planning']:
        sim_data = get_sim_data(task, 'zh')
        human_data = get_human_data(task)
        if task is None:
            scenario = 'All'
        else:
            scenario = task.split()
            scenario = [e.capitalize() for e in scenario]
            scenario = ' '.join(scenario)
        sim_stats = statistics(sim_data)
        human_stats = statistics(human_data)
        output = f'''\midrule
        \multirow{{2}}{{*}}{{{scenario}}} & Simu. & {sim_stats["convs"]} & {sim_stats["turns"]:.2f} & {sim_stats["avg_user_tokens"]:.2f} & {sim_stats["avg_user_tokens_turn"]:.2f} & {sim_stats["avg_assistant_tokens_turn"]:.2f} \\\\
        & Human & {human_stats["convs"]} & {human_stats["turns"]:.2f} & {human_stats["avg_user_tokens"]:.2f} & {human_stats["avg_user_tokens_turn"]:.2f} & {human_stats["avg_assistant_tokens_turn"]:.2f} \\\\
        '''
        print(output)
        while input('Continue? (y/n) ') != 'y':
            pass
