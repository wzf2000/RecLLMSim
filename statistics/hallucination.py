import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from utils import SIM_DIR, HUMAN_DIR, HUMAN_DIR_V2

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


def get_sim_hallucination(tasks: list[str] | None = None) -> tuple[list[int], list[int]]:
    if tasks is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    conv_ret = []
    utt_ret = []
    for task in tasks:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(SIM_DIR, task, file), 'r') as f:
                data = json.load(f)
            history = data['history']
            flag = False
            for utt in history:
                if 'hallucination' in utt:
                    if utt['hallucination']['hallucination']:
                        utt_ret.append(1)
                        flag = True
                    else:
                        utt_ret.append(0)
            if flag:
                conv_ret.append(1)
            else:
                conv_ret.append(0)
    return conv_ret, utt_ret

def get_human_hallucination(tasks: list[str] | None = None) -> tuple[list[int], list[int]]:
    if tasks is None:
        tasks = ['preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    conv_ret = []
    utt_ret = []

    def update(dir: str):
        users = os.listdir(dir)
        for user in users:
            for task in tasks:
                if task not in task_translation_reverse:
                    continue
                task = task_translation_reverse[task]
                if not os.path.exists(os.path.join(dir, user, task)):
                    continue
                files = os.listdir(os.path.join(dir, user, task))
                files = [file for file in files if file.endswith('.json')]
                files.sort(key=lambda x: x.split('.')[0])
                for file in files:
                    with open(os.path.join(dir, user, task, file), 'r') as f:
                        data = json.load(f)
                    history = data['history']
                    flag = False
                    for utt in history:
                        if 'hallucination' in utt:
                            if utt['hallucination'] == '是':
                                utt_ret.append(1)
                                flag = True
                            else:
                                utt_ret.append(0)
                    if flag:
                        conv_ret.append(1)
                    else:
                        conv_ret.append(0)
    update(HUMAN_DIR)
    update(HUMAN_DIR_V2)
    return conv_ret, utt_ret

if __name__ == '__main__':
    row_names = ['All', 'Travel', 'Recipe', 'Gifts', 'Skills']
    conv_data = {'Human': {}, 'Agent': {}}
    utt_data = {'Human': {}, 'Agent': {}}
    for row_name, tasks in zip(row_names, [['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning'], ['new travel planning', 'travel planning'], ['recipe planning'], ['preparing gifts'], ['skills learning planning']]):
        conv_data['Human'][row_name], utt_data['Human'][row_name] = get_human_hallucination(tasks)
        conv_data['Agent'][row_name], utt_data['Agent'][row_name] = get_sim_hallucination(tasks)
    # plot a Figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 2))
    for i, (data, title) in enumerate(zip([conv_data, utt_data], ['Conversation Hallucination', 'Utterance Hallucination'])):
        ax = axs[i]
        labels = list(data['Human'].keys())
        human_data = [data['Human'][label] for label in labels]
        agent_data = [data['Agent'][label] for label in labels]

        human_percentages = [np.mean(human) * 100 for human in human_data]
        agent_percentages = [np.mean(agent) * 100 for agent in agent_data]

        x = np.arange(len(labels))
        width = 0.35  # the width of the bars
        ax.bar(x - width/2, human_percentages, width, label='Human', color='blue')
        ax.bar(x + width/2, agent_percentages, width, label='Agent', color='orange')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 20)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'hallucination_awareness.pdf'))
    plt.show()
