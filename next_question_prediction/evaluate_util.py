import numpy as np
from sklearn import metrics


def compute_classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
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
