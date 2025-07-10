import numpy as np
from typing import Iterable, Sequence
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report
from transformers.trainer_utils import EvalPrediction

def evaluate(predictions: Sequence[int | float | np.integer], ground_truth: Sequence[int | np.integer], binary: bool = False) -> tuple[float, float]:
    classfication = all(isinstance(p, (int, np.integer)) for p in predictions)
    # output the index that gt != prediction
    # for i, (p, gt) in enumerate(zip(predictions, ground_truth)):
    #     if p != gt:
    #         print(f"Index: {i}, GT: {gt}, Prediction: {p}")
    predictions_arr = np.array(predictions)
    ground_truth_arr = np.array(ground_truth)
    if classfication:
        acc = accuracy_score(ground_truth_arr, predictions_arr)
        f1 = f1_score(ground_truth_arr, predictions_arr, average='weighted')
        print(f"Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")
        if not binary:
            print(classification_report(ground_truth_arr, predictions_arr, digits=4))
    mse = mean_squared_error(ground_truth_arr, predictions_arr)
    rmse = mse ** 0.5
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    print(f'GT mean score: {np.mean(ground_truth_arr):.4f}, Predicted mean score: {np.mean(predictions_arr):.4f}')
    return mse, rmse

def compute_metrics_cls(p: EvalPrediction) -> dict[str, float]:
    assert isinstance(p.predictions, np.ndarray)
    assert isinstance(p.label_ids, np.ndarray)
    logits = p.predictions
    labels = p.label_ids

    predictions = np.argmax(logits, axis=1)
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = mse ** 0.5
    predict_mean = np.mean(predictions)
    result = {
        'f1_weighted': f1,
        'accuracy': acc,
        'mse': mse,
        'rmse': rmse,
        'predict_mean': predict_mean
    }
    label_count = Counter(labels)
    if len(label_count) == 2:
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        recall_0 = tn / (tn + fp) if tn + fp > 0 else 0
        precision_0 = tn / (tn + fn) if tn + fn > 0 else 0
        recall_1 = tp / (tp + fn) if tp + fn > 0 else 0
        precision_1 = tp / (tp + fp) if tp + fp > 0 else 0
        result['recall_0'] = recall_0
        result['precision_0'] = precision_0
        result['recall_1'] = recall_1
        result['precision_1'] = precision_1
    return result

def compute_metrics_reg(p: EvalPrediction) -> dict[str, float]:
    assert isinstance(p.predictions, np.ndarray)
    assert isinstance(p.label_ids, np.ndarray)
    logits = p.predictions
    labels = p.label_ids
    mse = mean_squared_error(labels, logits)
    rmse = mse ** 0.5
    predict_mean = np.mean(logits)
    assert isinstance(predict_mean, float), f"Expected predict_mean to be float, got {type(predict_mean)}"
    return {
        'mse': mse,
        'rmse': rmse,
        'predict_mean': predict_mean
    }
