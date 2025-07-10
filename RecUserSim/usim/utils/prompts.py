import pandas as pd
import random
import re
import os
from typing import Tuple
from loguru import logger

from usim import DATA_PATH

def _get_task_context(file_name: str, sheet_name: str):
    task_context = pd.read_excel(file_name, sheet_name=sheet_name)
    return task_context

def _get_prompt_options(task_context: pd.DataFrame):
    Type = ""
    prompt_options = dict()
    for i in range(len(task_context)):
        if str(task_context['Type'][i]) != 'nan':
            Type = str(task_context['Type'][i])
            assert str(task_context['Prompt'][i]) != 'nan'
            prompt_options[Type] = {'Prompt': str(task_context['Prompt'][i])}
        Key = str(task_context['Key'][i])
        prompt_options[Type][Key] = []
        for option_id in range(1, 5):
            option = str(task_context['Option' + str(option_id)][i])
            if option != 'nan':
                prompt_options[Type][Key].append(option)
    return prompt_options

def _complete_with_choices(text: str, choices: dict):
    for key in choices:
        text = text.replace(f'[{key}]', choices[key])
        # find substr with format as [key:{"key1": "value1","key2":"value2"}]
        substrs = re.findall(r'\[' + key + r':\{.*?\}\]', text)
        for substr in substrs:
            replace_dict = eval(substr[substr.find('{'):-1])
            text = text.replace(substr, replace_dict[choices[key]])
    return text

def random_complete_prompt_with_type(prompt_options: dict, Type: str):
    if Type not in prompt_options:
        logger.warning('Type not found')
        return None
    options = prompt_options[Type]
    prompt = options['Prompt']
    keys = list(options.keys())
    keys.remove('Prompt')
    choices = {}
    for key in keys:
        choices[key] = _complete_with_choices(random.choice(options[key]), choices)
    prompt = _complete_with_choices(prompt, choices)
    return prompt

def all_complete_prompts_with_type(prompt_options: dict, Type: str):
    if Type not in prompt_options:
        logger.warning('Type not found')
        return None
    options = prompt_options[Type]
    prompt = options['Prompt']
    keys = list(options.keys())
    keys.remove('Prompt')

    def enumerate_options(enumerate_id: int, choices: dict):
        if enumerate_id == len(keys):
            return [_complete_with_choices(prompt, choices)]
        key = keys[enumerate_id]
        options = prompt_options[Type][key]
        prompts = []
        for option in options:
            choices[key] = _complete_with_choices(option, choices)
            prompts.extend(enumerate_options(enumerate_id + 1, choices))
        return prompts

    prompts = enumerate_options(0, {})
    return prompts

def _get_all_prompts_with_prompt_options(prompt_options: dict):
    prompts = {}
    for Type in prompt_options:
        prompts[Type] = all_complete_prompts_with_type(prompt_options, Type)
    return prompts

def get_all_prompts(file_name: str, sheet_name: str):
    task_context = _get_task_context(file_name, sheet_name)
    prompt_options = _get_prompt_options(task_context)
    prompts = _get_all_prompts_with_prompt_options(prompt_options)
    return prompts

def random_complete_prompt(file_name: str, sheet_name: str, exclude_prompt: list = []):
    prompts = get_all_prompts(file_name, sheet_name)
    # merge the prompts into one list
    all_prompts = []
    for Type in prompts:
        all_prompts.extend(prompts[Type])
    # remove the prompts in exclude_prompt
    for prompt in exclude_prompt:
        if prompt in all_prompts:
            all_prompts.remove(prompt)
    return random.choice(all_prompts)

def random_prompt_with_idx(file_name: str, sheet_name: str, exclude_prompt: list = [], exclude_type: list = []) -> Tuple[str, int, str]:
    prompts = get_all_prompts(file_name, sheet_name)
    # merge the prompts into one list
    all_prompts = []
    for Type in prompts:
        if Type in exclude_type:
            continue
        all_prompts.extend(prompts[Type])
    # random select a prompt from all_prompts and return the Type, index and prompt
    prompt = random.choice(all_prompts)
    while prompt in exclude_prompt:
        prompt = random.choice(all_prompts)
    for Type in prompts:
        if prompt in prompts[Type]:
            return Type, prompts[Type].index(prompt), prompt
    assert False, 'Prompt not found in prompt options'

if __name__ == '__main__':
    task_context = _get_task_context(os.path.join(DATA_PATH, 'task_context.xlsx'), 'sheet1')
    prompt_options = _get_prompt_options(task_context)
    logger.info(random_complete_prompt_with_type(prompt_options, '礼物准备'))
