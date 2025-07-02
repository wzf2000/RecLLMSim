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

rating_keys = ['Detail Level', 'Practical Usefulness', 'Diversity']
rating_keys_satisfaction = ['detail', 'utility', 'diversity']
rating_keys_mapping = {
    'Detail Level': 'Detail',
    'Practical Usefulness': 'Availability',
    'Diversity': 'Diversity'
}
rating_mapping = {
    '详细': 2,
    '一般': 1,
    '简略': 0,
    '很有用': 2,
    '有一定参考': 1,
    '不可用': 0,
    '结果多样': 2,
    '结果单一': 0,
}

def get_sim_quality(tasks: list[str] | None = None, self_report: bool = False) -> dict[str, list[int]]:
    if tasks is None:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    quality = {key: [] for key in rating_keys}
    for task in tasks:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(SIM_DIR, task, file), 'r') as f:
                data = json.load(f)
            if self_report:
                for origin_key, key in zip(rating_keys, rating_keys_satisfaction):
                    quality[origin_key].append(data['satisfaction'][key])
            else:
                for key in rating_keys:
                    quality[key].append(data['rating'][key])
    return quality

def get_human_quality(tasks: list[str] | None = None) -> dict[str, list[int]]:
    if tasks is None:
        tasks = ['preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    quality = {key: [] for key in rating_keys}

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
                    for key in rating_keys:
                        summary = rating_keys_mapping[key]
                        for question in data['questionnaire']:
                            if question['summary'] == summary:
                                quality[key].append(rating_mapping[question['option']])
                                break
    update(HUMAN_DIR)
    update(HUMAN_DIR_V2)
    print(f"Quality data: {len(quality[rating_keys[0]])} samples")
    return quality

def get_human_quality_by_model() -> dict[str, dict[str, list[int]]]:
    tasks = ['preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    quality = {key: {} for key in rating_keys}

    def update(dir: str):
        users = os.listdir(dir)
        for user in users:
            for task in tasks:
                task = task_translation_reverse[task]
                if not os.path.exists(os.path.join(dir, user, task)):
                    continue
                files = os.listdir(os.path.join(dir, user, task))
                files = [file for file in files if file.endswith('.json')]
                files.sort(key=lambda x: x.split('.')[0])
                for file in files:
                    with open(os.path.join(dir, user, task, file), 'r') as f:
                        data = json.load(f)
                    chat_model = data['chat_model'] if 'chat_model' in data else 'gpt-4-turbo-preview'
                    for key in rating_keys:
                        summary = rating_keys_mapping[key]
                        for question in data['questionnaire']:
                            if question['summary'] == summary:
                                if chat_model not in quality[key]:
                                    quality[key][chat_model] = []
                                quality[key][chat_model].append(rating_mapping[question['option']])
                                break
    update(HUMAN_DIR)
    update(HUMAN_DIR_V2)
    return quality

def plot(ax: plt.Axes, labels: list[str], sizes: list[int], title: str, legend: bool):
    patches, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '',
        startangle=90,
        textprops={'fontsize': 25},
        radius=1.2  # 增大饼图半径
    )
    if legend:
        ax.legend(
            handles=patches,  # 使用扇形对象作为图例句柄
            labels=labels,    # 自定义标签文本
            title="Scores",
            loc="upper right",
            bbox_to_anchor=(1.08, 1.2),  # 将图例放在图表右侧
            fontsize=20,
            title_fontsize=25,
        )

    # 为未显示的小扇区添加注释
    threshold = 5
    for i, patch in enumerate(patches):
        percent = sizes[i] / sum(sizes) * 100
        if percent < threshold:
            # 计算扇区中心角度
            ang = (patch.theta2 + patch.theta1) / 2
            # 计算引导线终点
            x = 1.3 * np.cos(np.deg2rad(ang))
            y = 1.3 * np.sin(np.deg2rad(ang))
            # 绘制引导线
            ax.annotate(
                f'{percent:.1f}%',
                xy=(np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))),
                xytext=(x, y),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3"),
                ha='left' if ang < 180 else 'right',
                va='center',
                fontsize=18
            )

    ax.axis('equal')
    ax.set_title(title, fontsize=30, fontweight='bold')

if __name__ == '__main__':
    for tasks in [['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning'], ['new travel planning', 'travel planning'], ['recipe planning'], ['preparing gifts'], ['skills learning planning']]:
        quality_data = get_sim_quality(tasks, self_report=True)
        # quality_data = get_human_quality(tasks)
        # plot subplots
        fig, axs = plt.subplots(1, len(rating_keys), figsize=(15, 6))
        output = ''
        for i, key in enumerate(rating_keys):
            # plot pie chart
            counter = Counter(quality_data[key])
            total = sum(counter.values())
            for rating in [0, 1, 2]:
                output += f'{counter[rating] / total:.2%}\t'
            avg_rating = sum(rating * counter[rating] for rating in [0, 1, 2]) / total
            output += f'{avg_rating:.3f}\t'
            # labels = list(counter.keys())
            # sizes = list(counter.values())
            # plot(axs[i], labels, sizes, key, legend=(i == len(rating_keys) - 1))
        output = output.strip()
        print(output)
        # plt.tight_layout()
        # plt.savefig(os.path.join('figures', 'assistant_quality.pdf'))
    # quality_data_by_model = get_human_quality_by_model()
    # Output the mean scores of each model for each rating
    # Latex Table
    # print("\\begin{table}[htbp]")
    # print("\\centering")
    # print('\\begin{threeparttable}[c]')
    # print("\\begin{tabular}{|c|c|c|c|}")
    # print("\\toprule")
    # print("\\textbf{Model} & \\textbf{Detail Level} & \\textbf{Practical Usefulness} & \\textbf{Diversity} \\\\")
    # print("\\midrule")
    # models = quality_data_by_model[rating_keys[0]].keys()
    # for model in models:
    #     print(f"\\textbf{{{model}}} & ", end="")
    #     for key in rating_keys:
    #         mean = np.mean(quality_data_by_model[key][model])
    #         print(f"{mean:.4f} & ", end="")
    #     print("\\\\")
    # print('\\bottomrule')
    # print("\\end{tabular}")
    # print("\\end{threeparttable}")
    # print("\\caption{Quality of human assistant responses by model}")
    # print("\\label{tab:assistant_quality_by_model}")
    # print("\\end{table}")
