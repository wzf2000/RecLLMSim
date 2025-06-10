import os
import torch
import random
import numpy as np

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')
HUMAN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'human_exp_V2')

def conv_format(history: list[dict[str, str]]) -> str:
    text = ''
    for turn in history:
        if turn['role'] == 'user':
            text += f'User: {turn["content"]}\n\n'
        else:
            text += f'System: {turn["content"]}\n\n'
    return text

def get_profile(profile: dict) -> str:
    return f"""性别：{'女' if profile['gender'] == 'Female' else '男'}
年龄：{profile['age']}
职业：{profile['occupation']}
职业背景：{profile['background']}
性格：{'，'.join(profile['personality'])}
日常兴趣爱好：{'，'.join(profile['daily_interests'])}
旅行习惯：{'，'.join(profile['travel_habits'])}
饮食偏好：{'，'.join(profile['dining_preferences'])}
消费习惯：{'，'.join(profile['spending_habits'])}
其他方面：{'，'.join(profile['other_aspects'])}
"""

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.empty_cache()
