import json
import os
import random
from loguru import logger

from usim import POOL_PATH, POOL_NAME, CONFIG_PATH, POOL_SIZE
from usim.chat import get_chatbot, single_turn_chat

def count_sample(pool: str):
    count = 0
    while f'{count + 1}.' in pool:
        count += 1
    return count

def build_preference_pool(prompt: str, cont: str, version: int, pool_size: int = 50, model: str = None):
    if model is not None:
        chatbot = get_chatbot(model=model)
    else:
        chatbot = get_chatbot()
    prompt = prompt.replace(POOL_SIZE, str(pool_size))
    cont = cont.replace(POOL_SIZE, str(pool_size))
    logger.info(f'Prompt = {prompt}')
    logger.info(f'Continue = {cont}')
    ret = ''
    cnt = 0
    flag = True
    iter = 0
    while cnt < pool_size and flag:
        ret += '\n' + single_turn_chat(chatbot, 1, cont, output=False)
        cnt = count_sample(ret)
        iter += 1
        logger.info(f'[Iter {iter}]: 当前已生成{cnt}个用户画像')
        print('Set flag to False to stop the loop')
        breakpoint()
    with open(os.path.join(POOL_PATH, POOL_NAME.format(version)), "w") as f:
        f.write(ret)
    return ret

def build_new_preference_pool(file_name: str, pool_size: int = 50):
    with open(file_name, 'r') as f:
        data = json.load(f)
    prompt = data['prompt']
    cont = data['continue']
    # check data/pool directory exists
    # if not, create it
    if not os.path.exists(POOL_PATH):
        os.makedirs(POOL_PATH)
        version = 1
    else:
        dir_list = os.listdir(POOL_PATH)
        version = 1
        while POOL_NAME.format(version) in dir_list:
            version += 1
    if 'model' not in data:
        return build_preference_pool(prompt, cont, version, pool_size)
    return build_preference_pool(prompt, cont, version, pool_size, data['model'])

def random_sample_preference(file_name: str, sample_num: int = 1):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        data = ''.join(lines)
        preferences = data.split('\n\n')
    return random.sample(preferences, sample_num)

def get_all_preference(file_name: str):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        data = ''.join(lines)
        preferences = data.split('\n\n')
    return preferences

if __name__ == '__main__':
    # build_new_preference_pool(os.path.join(CONFIG_PATH, 'preference_gen.json'), 5)
    print(random_sample_preference(os.path.join(POOL_PATH, 'preference_pool.txt'))[0])
