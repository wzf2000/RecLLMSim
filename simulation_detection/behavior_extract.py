import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from data_util import get_human_data_dict, get_sim_data_dict, SIM_DIR_V2


class BehavioralFeatureExtractor:
    def __init__(self):
        self.behavioral_patterns = {
            'hesitation': [r'嗯+', r'啊+', r'那个+', r'就是+', r'可能', r'好像'],
            'correction': [r'不对', r'纠正一下', r'重新说', r'应该是'],
            'politeness': [r'请问', r'谢谢', r'感谢', r'抱歉', r'不好意思', r'您'],
            'uncertainty': [r'可能', r'也许', r'大概', r'不太确定', r'我觉得'],
            'engagement': [r'你呢？', r'你觉得', r'你怎么看', r'对吧？']
        }

    def extract_turn_features(self, turns: list[dict]) -> dict:
        """提取单轮对话的行为特征"""
        user_turns = [turn['content'] for turn in turns if turn['role'] == 'user']

        features = {}

        # 1. 对话节奏特征
        features['avg_turn_length'] = np.mean([len(turn) for turn in user_turns]) if user_turns else 0
        features['turn_length_variance'] = np.var([len(turn) for turn in user_turns]) if len(user_turns) > 1 else 0

        # 2. 行为模式特征
        all_user_text = ' '.join(user_turns)
        for pattern_type, patterns in self.behavioral_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, all_user_text))
            features[f'{pattern_type}_count'] = count

        # 3. 对话结构特征
        features['question_ratio'] = len(
            [t for t in user_turns if '？' in t or '?' in t]
        ) / len(user_turns) if user_turns else 0
        features['response_variety'] = len(
            set(user_turns)
        ) / len(user_turns) if user_turns else 0

        # 4. 语言复杂度特征
        features['avg_sentence_complexity'] = self.calculate_complexity(
            user_turns
        )

        return features

    def calculate_complexity(self, turns: list[str]) -> float:
        """计算语言复杂度"""
        if not turns:
            return 0

        complexities = []
        for turn in turns:
            # 句子数量
            sentences = re.split(r'[。！？!?]', turn)
            sentences = [s for s in sentences if s.strip()]
            if not sentences:
                continue

            # 平均句子长度
            avg_sent_len = np.mean([len(s) for s in sentences])

            # 词汇丰富度（简单版）
            words = re.findall(r'[\u4e00-\u9fa5]+', turn)
            if len(words) > 0:
                vocab_richness = len(set(words)) / len(words)
            else:
                vocab_richness = 0

            complexity = 0.5 * avg_sent_len + 0.5 * vocab_richness
            complexities.append(complexity)

        return np.mean(complexities) if complexities else 0

    def extract_session_features(self, dialogue: dict) -> dict:
        """提取整个对话会话的特征"""
        turns = dialogue['history']
        base_features = self.extract_turn_features(turns)

        # 6. 对话发展特征
        base_features['topic_development'] = self.analyze_topic_development(
            turns
        )

        return base_features

    def analyze_topic_development(self, turns: list[dict]) -> float:
        """分析话题发展情况"""
        if len(turns) < 3:
            return 0.3

        # 简单的主题延续性分析
        topic_changes = 0
        for i in range(len(turns) - 1):
            next_turn = turns[i + 1]['content']

            # 检查是否有明显的话题转换词
            if re.search(r'另外|对了|话说回来|换个话题', next_turn):
                topic_changes += 1

        # 适度的话题变化是自然的
        change_ratio = topic_changes / len(turns)
        if 0.1 <= change_ratio <= 0.3:
            return 0.8
        else:
            return 0.3

def get_features(dialogues: list[dict]) -> pd.DataFrame:
    extractor = BehavioralFeatureExtractor()
    features_list = []
    for dialogue in dialogues:
        features = extractor.extract_session_features(dialogue)
        features_list.append(features)
    return pd.DataFrame(features_list)

# 使用行为特征训练分类器

def train_behavioral_classifier(dialogues: list[dict], labels: list[int]) -> tuple:
    feature_df = get_features(dialogues)

    # 特征选择和标准化
    X = feature_df.values
    y = np.array(labels)

    # 选择最重要的k个特征
    selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 训练模型
    model = LogisticRegression(
        penalty='l1', solver='liblinear', random_state=42)
    model.fit(X_scaled, y)

    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_df.columns[selector.get_support()],
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)

    # sim 对话准确率
    sim_accuracy = model.score(X_scaled, y)
    print(f"Behavioral Classifier Simulation Detection Accuracy: {sim_accuracy:.2%}")

    # 画出 AUROC 图
    probs = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('tmp/behavioral_classifier_roc.png')

    return model, scaler, selector, feature_importance, feature_df.columns[selector.get_support()]

def plot_probability_histogram(model: LogisticRegression, scaler: StandardScaler, selector: SelectKBest, dialogues: list[dict], filename: str, sim: bool):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    feature_df = get_features(dialogues)
    X = feature_df.values
    X_selected = selector.transform(X)
    X_scaled = scaler.transform(X_selected)
    y_pred = model.predict(X_scaled)
    y_true = np.array([0 if sim else 1] * len(dialogues))
    print(f"Behavioral Classifier Accuracy on {'Simulated' if sim else 'Human'} Data: {(y_pred == y_true).mean():.2%}")
    probs = model.predict_proba(X_scaled)[:, 1]
    if sim:
        # 输出大于0.2的数量和占比
        print(f"    Proportion of dialogues with predicted probability > 0.2: {(probs > 0.2).mean():.2%}")
        print(f"    Number of dialogues with predicted probability > 0.2: {(probs > 0.2).sum()} out of {len(probs)}")
        # output_file = filename.replace('.png', '.json')
        # dialog_file_paths = [dialogue['file_path'] for i, dialogue in enumerate(dialogues)]
        # # 转为绝对路径
        # dialog_file_paths = [os.path.abspath(path) for path in dialog_file_paths]
        # file_prob_dict = {
        #     dialog_file_paths[i]: float(probs[i]) for i in range(len(dialogues))
        # }
        # with open(output_file, 'w') as f:
        #     json.dump(file_prob_dict, f, ensure_ascii=False, indent=4)

    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, alpha=0.7, color='green')
    plt.title('Histogram of Predicted Probabilities (Behavioral Classifier)')
    plt.xlabel('Predicted Probability of Being Simulated')
    plt.ylabel('Frequency')
    plt.savefig(filename)

def plot_feature_distributions(data_list: dict[str, list[dict]], filename: str, feature_name: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, dialogues in data_list.items():
        features = BehavioralFeatureExtractor()
        feature_values = []
        for dialogue in dialogues:
            feat = features.extract_session_features(dialogue)
            feature_values.append(feat[feature_name])
        # 纵坐标用百分比
        plt.hist(feature_values, bins=30, alpha=0.5, label=label, density=True)
    plt.title(f'Distribution of Feature: {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)

def main():
    sim_data = get_sim_data_dict()
    human_data = get_human_data_dict()
    new_sim_data = get_sim_data_dict(SIM_DIR_V2)
    new_sim_data_rewritten = get_sim_data_dict(SIM_DIR_V2, rewritten=True)
    dialogues = sim_data + human_data + new_sim_data + new_sim_data_rewritten
    dialogue_dict = {
        'human': human_data,
        'gpt-3.5/4': sim_data,
        'gpt-5': new_sim_data,
        'gpt-5_rewritten': new_sim_data_rewritten
    }
    labels = [0] * len(sim_data) + [1] * len(human_data) + [0] * len(new_sim_data) + [0] * len(new_sim_data_rewritten)
    print(f"Total dialogues: {len(dialogues)} (Simulated: {len(sim_data)} + {len(new_sim_data)} + {len(new_sim_data_rewritten)}, Human: {len(human_data)})")
    # 训练行为分类器
    model, scaler, selector, importance, selected_features = train_behavioral_classifier(dialogues, labels)
    # 找出最重要的三个特征
    k = 10
    top_k_features = importance.head(k)
    for feature in top_k_features['feature']:
        plot_feature_distributions(
            dialogue_dict,
            f'tmp/feature/distribution_{feature}.png',
            feature
        )
    plot_probability_histogram(
        model, scaler, selector, sim_data,
        'tmp/probability/sim.png', sim=True
    )
    plot_probability_histogram(
        model, scaler, selector, human_data,
        'tmp/probability/human.png', sim=False
    )
    plot_probability_histogram(
        model, scaler, selector, new_sim_data,
        'tmp/probability/new_sim.png', sim=True
    )
    plot_probability_histogram(
        model, scaler, selector, new_sim_data_rewritten,
        'tmp/probability/new_sim_rewritten.png', sim=True
    )

    print("最重要的行为特征:")
    print(importance)

    print("\n选中的特征:", selected_features.tolist())


if __name__ == "__main__":
    main()
