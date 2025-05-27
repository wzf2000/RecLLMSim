import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import Counter

from read_profile import get_sim_profile, get_human_profile
from utils import HUMAN_DIR, HUMAN_DIR_V2

# plt.rcParams['font.family'] = 'Times New Roman'

df1 = get_sim_profile()
df2 = get_human_profile(HUMAN_DIR)
df3 = get_human_profile(HUMAN_DIR_V2)
df2 = pd.concat([df2, df3], ignore_index=True)

features = ['daily_interests', 'travel_habits', 'dining_preferences', 'spending_habits']
feature_labels = ['Daily Interests', 'Travel Habits', 'Dining Preferences', 'Spending Habits']

def analyze_multilabel_distribution(series: pd.Series) -> Counter:
    """分析多标签特征的分布"""
    # 展平所有标签并计数
    all_tags = [tag for tags in series for tag in tags if tag != 'Others']
    return Counter(all_tags)

def plot_tag_distribution(ax: plt.Axes, counter1: Counter, counter2: Counter, title: str, legend_loc: str):
    """绘制标签分布对比图"""
    # 获取所有唯一标签
    all_tags = sorted(set(list(counter1.keys()) + list(counter2.keys())))

    # 输出 counter1 与 counter2 top5 的重合度
    top5_counter1 = counter1.most_common(5)
    top5_counter2 = counter2.most_common(5)
    top5_tags1 = [tag for tag, _ in top5_counter1]
    top5_tags2 = [tag for tag, _ in top5_counter2]
    overlap = len(set(top5_tags1) & set(top5_tags2))
    print(f"Top 5 overlap: {overlap} / 5")

    # 准备数据
    group1_counts = [counter1.get(tag, 0) for tag in all_tags]
    group2_counts = [counter2.get(tag, 0) for tag in all_tags]

    # 设置条形图
    x = np.arange(len(all_tags))
    width = 0.35

    ax.bar(x - width / 2, group1_counts, width, label='Simulation', color='lightcoral')
    ax.bar(x + width / 2, group2_counts, width, label='Human', color='lightblue')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_tags, rotation=30, ha='right')
    legend = ax.legend(loc=legend_loc)
    for text in legend.get_texts():
        text.set_fontsize(13)
        text.set_fontweight('bold')

# 创建可视化
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
interests_dist1 = analyze_multilabel_distribution(df1['daily_interests'])
interests_dist2 = analyze_multilabel_distribution(df2['daily_interests'])
plot_tag_distribution(plt.gca(), interests_dist1, interests_dist2, 'Daily Interests Distribution', 'upper right')

plt.subplot(1, 4, 2)
dining_dist1 = analyze_multilabel_distribution(df1['dining_preferences'])
dining_dist2 = analyze_multilabel_distribution(df2['dining_preferences'])
plot_tag_distribution(plt.gca(), dining_dist1, dining_dist2, 'Dining Preferences Distribution', 'upper center')

plt.subplot(1, 4, 3)
travel_dist1 = analyze_multilabel_distribution(df1['travel_habits'])
travel_dist2 = analyze_multilabel_distribution(df2['travel_habits'])
plot_tag_distribution(plt.gca(), travel_dist1, travel_dist2, 'Travel Habits Distribution', 'upper right')

plt.subplot(1, 4, 4)
spending_dist1 = analyze_multilabel_distribution(df1['spending_habits'])
spending_dist2 = analyze_multilabel_distribution(df2['spending_habits'])
plot_tag_distribution(plt.gca(), spending_dist1, spending_dist2, 'Spending Habits Distribution', 'upper center')

def create_co_occurrence_matrix(series: pd.Series, feature_name: str) -> pd.DataFrame:
    """创建标签共现矩阵"""
    all_tags = set([tag for tags in series for tag in tags])
    matrix = pd.DataFrame(0, index=list(all_tags), columns=list(all_tags))

    for tags in series:
        for tag1 in tags:
            for tag2 in tags:
                matrix.loc[tag1, tag2] += 1

    return matrix

# plt.subplot(2, 3, 2)
# co_occurrence1 = create_co_occurrence_matrix(df1['travel_habits'], 'Travel Habits')
# sns.heatmap(co_occurrence1, annot=True, cmap='YlOrRd')
# plt.title('Simulation: Travel Habits Co-occurrence')
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)

# plt.subplot(2, 3, 5)
# co_occurrence2 = create_co_occurrence_matrix(df2['travel_habits'], 'Travel Habits')
# sns.heatmap(co_occurrence2, annot=True, cmap='YlOrRd')
# plt.title('Human: Travel Habits Co-occurrence')
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)

# gender distribution

# plt.subplot(3, 4, 5)
# gender_counts = df1['gender'].value_counts()
# plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
# plt.title('Simulation Gender Distribution')

# plt.subplot(3, 4, 6)
# gender_counts = df2['gender'].value_counts()
# plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
# plt.title('Human Gender Distribution')

# # age distribution

def categorize_age(age: int) -> str:
    age_splits = [18, 26, 36, 50, 66, 100]
    age_labels = ['0-18', '19-26', '27-36', '37-50', '51-66', '67+']
    for i, split in enumerate(age_splits):
        if age < split:
            return age_labels[i]

df1['age'] = df1['age'].apply(lambda x: int(x))
df1['age_categorized'] = df1['age'].apply(categorize_age)
df2['age'] = df2['age'].apply(lambda x: int(x))
df2['age_categorized'] = df2['age'].apply(categorize_age)

# plt.subplot(3, 4, 7)
# age_counts = df1['age_categorized'].value_counts()
# plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%')
# plt.title('Simulation Age Distribution')

# plt.subplot(3, 4, 8)
# age_counts = df2['age_categorized'].value_counts()
# plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%')
# plt.title('Human Age Distribution')

def calculate_multilabel_diversity(series: pd.Series) -> dict:
    """计算多标签特征的多样性指数"""
    # 统计每个标签的出现次数
    counter = analyze_multilabel_distribution(series)
    total = sum(counter.values())

    # 计算每个标签的比例
    proportions = [count / total for count in counter.values()]

    # 计算Simpson和Shannon多样性指数
    simpson = 1 - sum([p ** 2 for p in proportions])
    shannon = entropy(proportions)

    return {'Simpson': simpson, 'Shannon': shannon}

def caclulate_diversity(series: pd.Series) -> dict:
    """计算单标签特征的多样性指数"""
    propotions = series.value_counts(normalize=True)

    simpson = 1 - sum([p ** 2 for p in propotions])
    shannon = entropy(propotions)
    return {'Simpson': simpson, 'Shannon': shannon}

def calculate_all_diversity_scores(df: pd.DataFrame) -> dict:
    scores = {
        'age': caclulate_diversity(df['age_categorized']),
    }
    for column in features + ['other_aspects']:
        scores[column] = calculate_multilabel_diversity(df[column])
    return scores

diversity_scores1 = calculate_all_diversity_scores(df1)
diversity_scores2 = calculate_all_diversity_scores(df2)

# plt.subplot(3, 2, 5)
# x = np.arange(len(features) + 2)
# width = 0.35

# simpson_scores1 = [score['Simpson'] for score in diversity_scores1.values()]
# simpson_scores2 = [score['Simpson'] for score in diversity_scores2.values()]

# plt.bar(x - width / 2, simpson_scores1, width, label='Simulation', color='lightcoral')
# plt.bar(x + width / 2, simpson_scores2, width, label='Human', color='lightblue')
# plt.ylim(0, 1)
# plt.xticks(x, ['Age'] + feature_labels + ['Other Aspects'])
# plt.title('Simpson Diversity Score Comparison')
# plt.legend(loc='upper right')

# plt.subplot(3, 2, 6)
# x = np.arange(len(features) + 2)
# width = 0.35

# shannon_scores1 = [score['Shannon'] for score in diversity_scores1.values()]
# shannon_scores2 = [score['Shannon'] for score in diversity_scores2.values()]

# plt.bar(x - width / 2, shannon_scores1, width, label='Simulation', color='lightcoral')
# plt.bar(x + width / 2, shannon_scores2, width, label='Human', color='lightblue')
# plt.ylim(0, 3)
# plt.xticks(x, ['Age'] + feature_labels + ['Other Aspects'])
# plt.title('Shannon Diversity Score Comparison')
# plt.legend(loc='upper right')

plt.tight_layout(pad=0.3)
plt.savefig('figures/profile.pdf')

# 打印详细的统计信息
# print("\nDetailed Statistics:")
# for feature, feature_label in zip(features, feature_labels):
#     print(f"\n{feature_label} Distribution:")
#     print("\nSimulation:")
#     print(analyze_multilabel_distribution(df1[feature]))
#     print("\nHuman:")
#     print(analyze_multilabel_distribution(df2[feature]))

print("\nDiversity Scores:")
for feature, feature_label in zip(features, feature_labels):
    print(f"\n{feature_label}:")
    print(f"Simulation - Simpson: {diversity_scores1[feature]['Simpson']:.3f}, "
          f"Shannon: {diversity_scores1[feature]['Shannon']:.3f}")
    print(f"Human - Simpson: {diversity_scores2[feature]['Simpson']:.3f}, "
          f"Shannon: {diversity_scores2[feature]['Shannon']:.3f}")
