import numpy as np
from sklearn import metrics

def hit_rate(labels: np.ndarray, probs: np.ndarray, k: int):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    hits = 0
    for i in range(len(labels)):
        label_set = np.where(labels[i] == 1)[0]
        for label in label_set:
            if label in top_k[i]:
                hits += 1
                break
    return hits / len(labels)

def recall(labels: np.ndarray, probs: np.ndarray, k: int):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    recalls = 0
    total = 0
    for i in range(len(labels)):
        label_set = np.where(labels[i] == 1)[0]
        recalls += len(set(label_set) & set(top_k[i]))
        total += len(label_set)
    return recalls / total

def compute_classification_metrics(labels: np.ndarray, preds: np.ndarray):
    f1_micro_average = metrics.f1_score(labels, preds, average='micro')
    f1_macro_average = metrics.f1_score(labels, preds, average='macro')
    f1_weighted_average = metrics.f1_score(labels, preds, average='weighted')
    accuracy = metrics.accuracy_score(labels, preds)
    return {
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'f1_weighted': f1_weighted_average,
        'accuracy': accuracy,
    }

def compute_metrics(labels: np.ndarray, probs: np.ndarray, ranking_only: bool = False):
    preds = np.where(probs >= 0.5, 1, 0)
    # 计算各项指标
    if not ranking_only:
        f1_micro_average = metrics.f1_score(labels, preds, average='micro')
        f1_macro_average = metrics.f1_score(labels, preds, average='macro')
        f1_weighted_average = metrics.f1_score(labels, preds, average='weighted')
        accuracy = metrics.accuracy_score(labels, preds)

    # 计算hitrate与recall
    hit_rate_1 = hit_rate(labels, probs, 1)
    hit_rate_3 = hit_rate(labels, probs, 3)
    recall_1 = recall(labels, probs, 1)
    recall_3 = recall(labels, probs, 3)

    if ranking_only:
        return {
            'hit_rate_1': hit_rate_1,
            'hit_rate_3': hit_rate_3,
            'recall_1': recall_1,
            'recall_3': recall_3,
        }
    return {
        'f1_micro': f1_micro_average,
        'f1_macro': f1_macro_average,
        'f1_weighted': f1_weighted_average,
        'accuracy': accuracy,
        'hit_rate_1': hit_rate_1,
        'hit_rate_3': hit_rate_3,
        'recall_1': recall_1,
        'recall_3': recall_3,
    }
