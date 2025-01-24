import os
import json
import matplotlib.pyplot as plt

from basic_info import HUMAN_DIR, OUTPUT_DIR

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

dir_name = HUMAN_DIR
file_dir = OUTPUT_DIR
max_freq = {}
users = os.listdir(dir_name)
freq_dict = {}
# strategy_field = 'strategy'
# key_value = ['planning_Planning', 'context_High', 'question_Specific']
strategy_field = 'strategy_V2'
key_value = ['order_DepthBreadth', 'order_Breadth', 'feedback_NoFeedback', 'feedback_Negative']
for task_list in [['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']]:
    for user in users:
        counts = {}
        cnt = 0
        for task in task_list:
            if not os.path.exists(os.path.join(dir_name, user, task)):
                continue
            files = os.listdir(os.path.join(dir_name, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(dir_name, user, task, file), 'r') as f:
                    data = json.load(f)
                strategy = data[strategy_field]['final']
                for key in strategy:
                    if key not in counts:
                        counts[key] = {}
                    if strategy[key] not in counts[key]:
                        counts[key][strategy[key]] = 0
                    counts[key][strategy[key]] += 1
                cnt += 1
        for key in counts:
            if key not in max_freq:
                max_freq[key] = []
            max_count = 0
            for value in counts[key]:
                max_count = max(max_count, counts[key][value])
            max_freq[key].append(max_count / cnt)
        for key in key_value:
            if key not in freq_dict:
                freq_dict[key] = []
            key, value = key.split('_')
            freq_dict[f'{key}_{value}'].append(counts[key].get(value, 0) / cnt)

    for key in max_freq:
        print(key, sum(max_freq[key]) / len(max_freq[key]))
        # count the user that max_freq > 0.8
        print(len(max_freq[key]), len([x for x in max_freq[key] if x > 0.8]))
    for key in freq_dict:
        # plot the distribution of the key
        plt.hist(freq_dict[key], bins=10, range=[0, 1], edgecolor='white')
        plt.xlabel(f'Frequency of {key.replace("_", " ")}')
        plt.ylabel('User Count')
        plt.title(f'Distribution of {key.replace("_", " ")}')
        plt.savefig(os.path.join(file_dir, f'{key}.png'))
        plt.cla()
