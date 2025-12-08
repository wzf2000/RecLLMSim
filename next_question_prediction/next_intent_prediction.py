import os
import json
import numpy as np
import xgboost as xgb
from loguru import logger
from typing import overload, Literal
from collections import Counter
from argparse import ArgumentParser, Namespace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification

from nqp_data import get_nqp_data, get_nqp_data_sim, get_nqp_data_sim_rewritten, get_task_list
from evaluate_util import compute_classification_metrics
from ml_pipe import MLModel
from lm_pipe import get_dataset, get_trainer
from llm_pipe import predict_intents
from utils import SIM_DIR, SIM_DIR_V2


def process_data(train_data: list[dict], test_data: list[dict], cut: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    le = LabelEncoder()
    all_labels = [item['intent'] for item in train_data + test_data]
    le.fit(all_labels)
    y_all = le.transform(all_labels)
    X_train = np.array([item['history'] if not cut else item['cut_history'] for item in train_data])
    y_train = y_all[:len(train_data)]
    X_test = np.array([item['history'] if not cut else item['cut_history'] for item in test_data])
    y_test = y_all[len(train_data):]
    return X_train, y_train, X_test, y_test

def filter_data(sim_data: list[dict]) -> list[dict]:
    with open('../simulation_detection/tmp/probability_sim.json', 'r') as f:
        prob_dict = json.load(f)
    filtered_data = []
    for data in zip(sim_data):
        if prob_dict.get(os.path.abspath(data['file_path']), 0.0) >= 0.2:
            filtered_data.append(data)
    logger.info(f"Filtered simulated data size: {len(filtered_data)} out of {len(sim_data)}")
    return filtered_data

def get_augmentation(augmentation_desc: str, train_data: list[dict], data_task_name: str, filter: bool, data_version: int = 1) -> list[dict]:
    counter = Counter([item['intent'] for item in train_data])
    if data_version == 3:
        sim_data = get_nqp_data_sim_rewritten(task_list=get_task_list(human=False, task_name=data_task_name))
    else:
        sim_data = get_nqp_data_sim(task_list=get_task_list(human=False, task_name=data_task_name), sim_dir=SIM_DIR_V2 if data_version == 2 else SIM_DIR)
    if filter:
        sim_data = filter_data(sim_data)
    if '_' not in augmentation_desc:  # XXX: X times
        logger.info("Sample from all simulated data for augmentation")
    elif augmentation_desc.startswith('hot_'):  # hot_X_Y: use top X intents, Y times
        intent_num = int(augmentation_desc.split('_')[1])
        most_common = counter.most_common(intent_num)
        filtered_intents = set([item[0] for item in most_common])
        filtered_indices = [i for i, item in enumerate(sim_data) if item['intent'] in filtered_intents]
        sim_data = [sim_data[i] for i in filtered_indices]
        logger.info(f"Using simulated data for top {intent_num} intents for augmentation")
    elif augmentation_desc.startswith('cold_'):  # cold_X_Y: use bottom X intents, Y times
        intent_num = int(augmentation_desc.split('_')[1])
        least_common = counter.most_common()[:-intent_num-1:-1]
        filtered_intents = set([item[0] for item in least_common])
        filtered_indices = [i for i, item in enumerate(sim_data) if item['intent'] in filtered_intents]
        sim_data = [sim_data[i] for i in filtered_indices]
        logger.info(f"Using simulated data for bottom {intent_num} intents for augmentation")
    elif augmentation_desc.startswith('ratio_'):  # ratio_XXXX: use ratio XXXX of each intent
        ratio = float(augmentation_desc.split('_')[1])
        desired_counts = {label: int(count * ratio) for label, count in counter.items()}
        filtered_indices = []
        current_counts = Counter()
        # shuffle
        np.random.seed(42)
        perm = np.random.permutation(len(sim_data))
        sim_data = [sim_data[i] for i in perm]
        for i, item in enumerate(sim_data):
            if current_counts[item['intent']] < desired_counts.get(item['intent'], 0):
                filtered_indices.append(i)
                current_counts[item['intent']] += 1
        sim_data = [sim_data[i] for i in filtered_indices]
        logger.info(f"Using simulated data with ratio {ratio} for augmentation")
    else:
        raise NotImplementedError(f"Augmentation description {augmentation_desc} not implemented")
    aug_times = float(augmentation_desc.split('_')[-1])
    aug_samples = min(int(len(train_data) * aug_times), len(sim_data))
    if aug_samples == len(sim_data):
        logger.info("Using all filtered simulated data for augmentation")
    np.random.seed(42)
    selected_indices = np.random.choice(len(sim_data), size=aug_samples, replace=False)
    selected_sim_data = [sim_data[i] for i in selected_indices]
    return selected_sim_data

@overload
def get_data(task: str, data_task_name: str, filter: bool, val: Literal[False] = False, cut: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: ...

@overload
def get_data(task: str, data_task_name: str, filter: bool, val: Literal[True] = True, cut: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: ...

def get_data(task: str, data_task_name: str, filter: bool, val: bool = False, cut: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    train_data, test_data = get_nqp_data(sample=False, task_list=get_task_list(human=True, task_name=data_task_name))
    if val:
        train_indices, val_indices = train_test_split(range(len(train_data)), test_size=0.25, random_state=42)
    if task == 'human':
        aug_samples = 0
    elif task.startswith('aug_'):
        aug_data = get_augmentation(task[4:], train_data, data_task_name=data_task_name, filter=filter, data_version=1)
        train_data = aug_data + train_data
        aug_samples = len(aug_data)
        logger.info(f"After augmentation, total training samples: {len(train_data)}")
    elif task.startswith('augv2_'):
        aug_data = get_augmentation(task[6:], train_data, data_task_name=data_task_name, filter=filter, data_version=2)
        train_data = aug_data + train_data
        aug_samples = len(aug_data)
        logger.info(f"After augmentation v2, total training samples: {len(train_data)}")
    elif task.startswith('augv3_'):
        aug_data = get_augmentation(task[6:], train_data, data_task_name=data_task_name, filter=filter, data_version=3)
        train_data = aug_data + train_data
        aug_samples = len(aug_data)
        logger.info(f"After augmentation v3, total training samples: {len(train_data)}")
    else:
        raise NotImplementedError(f"Task {task} not implemented")
    X_train, y_train, X_test, y_test = process_data(train_data, test_data, cut=cut)
    if val:
        X_val = X_train[np.array(val_indices) + aug_samples]
        y_val = y_train[np.array(val_indices) + aug_samples]
        X_train = X_train[np.concatenate([np.array(train_indices) + aug_samples, np.arange(aug_samples)], axis=0)]
        y_train = y_train[np.concatenate([np.array(train_indices) + aug_samples, np.arange(aug_samples)], axis=0)]
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test, aug_samples
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_test, y_test, aug_samples

def check_log(log_file: str, model_name: str) -> bool:
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line.startswith(model_name + '\t'):
                logger.info(f"Model {model_name} already logged in {log_file}, skipping...")
                return True
    return False

def log_metrics(log_file: str, model_name: str, metrics: dict[str, float]):
    logger.info(f"Metrics: {metrics}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            keys = ['model'] + list(metrics.keys())
            f.write('\t'.join(keys) + '\n')
    with open(log_file, 'a') as f:
        values = [model_name] + [f'{metrics[k]:.4f}' for k in metrics.keys()]
        f.write('\t'.join(values) + '\n')

def work_ml(model_type: str, task: str, data_task_name: str, filter: bool, debug: bool, **kwargs) -> dict[str, float]:
    if not debug:
        log_file = os.path.join('results', data_task_name, f'intent-{task}.txt' if not filter else f'intent-{task}-filtered.txt')
        if check_log(log_file, model_type):
            return {}
    model = MLModel(model_type, **kwargs)
    X_train, y_train, X_test, y_test, aug_samples = get_data(task=task, data_task_name=data_task_name, filter=filter, cut=True)
    model.fit(X_train, y_train, aug_samples=aug_samples)
    logger.info(f"Model {model_type} trained.")
    preds = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, preds)
    if not debug:
        log_metrics(log_file, model_type, metrics)
    return metrics

def work_lm(model_name: str, task: str, data_task_name: str, filter: bool, debug: bool, **kwargs) -> dict[str, float]:
    if not debug:
        log_file = os.path.join('results', data_task_name, f'intent-{task}.txt' if not filter else f'intent-{task}-filtered.txt')
        if check_log(log_file, model_name):
            return {}
    X_train, y_train, X_val, y_val, X_test, y_test, _ = get_data(task=task, data_task_name=data_task_name, val=True, filter=filter)
    train_dataset, val_dataset, test_dataset = get_dataset(model_name, X_train, y_train, X_val, y_val, X_test, y_test)
    labels = list(set(y_train) | set(y_val) | set(y_test))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
    )
    trainer = get_trainer(
        model, train_dataset, val_dataset,
        output_dir=os.path.join(
            os.path.dirname(__file__),
            'results',
            data_task_name,
            'ckpts',
            f'intent-{task}' if not filter else f'intent-{task}-filtered'
        ),
        run_name=(
            f"intent/{data_task_name}/{task}/{model_name}" if not filter
            else f"intent/{data_task_name}/{task}-filtered/{model_name}"
        ),
        **kwargs
    )
    trainer.train()
    logger.info(f"Model {model_name} trained.")
    results = trainer.evaluate(test_dataset)
    metrics = {
        'f1_micro': results['eval_f1_micro'],
        'f1_macro': results['eval_f1_macro'],
        'f1_weighted': results['eval_f1_weighted'],
        'accuracy': results['eval_accuracy'],
    }
    if not debug:
        log_metrics(log_file, model_name, metrics)
    return metrics

def work_llm(model_name: str, task: str, data_task_name: str, max_workers: int, debug: bool, **kwargs) -> dict[str, float]:
    if not debug:
        log_file = os.path.join('results', data_task_name, f'intent-{task}.txt')
        if check_log(log_file, model_name):
            return {}
    train_data, test_data = get_nqp_data(sample=False, task_list=get_task_list(human=True, task_name=data_task_name))
    test_labels = np.array([item['intent'] for item in test_data])
    predicted_labels = predict_intents(model_name, test_data, max_workers=max_workers)
    metrics = compute_classification_metrics(test_labels, predicted_labels)
    if not debug:
        log_metrics(log_file, model_name, metrics)
    return metrics

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # global arguments
    parser.add_argument('-t', '--task', type=str, default='human', help='Task to run')
    parser.add_argument('--data_task_name', type=str, default='all', help='Task list for data loading: all or specific task name', choices=['all', 'travel', 'gift', 'recipe', 'skill'])
    parser.add_argument('--filter', action='store_true', help='Whether to filter out samples with low confidence predictions in simulated data')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode')

    # add sub-commands for type
    subparsers = parser.add_subparsers(dest='pipe', help='Sub-commands of method pipeline type: ml or lm', required=True)

    # Machine learning parser
    ml_parser = subparsers.add_parser('ml', help='Machine learning model')
    ml_parser.add_argument('-m', '--model_type', type=str, default='LR', help='Model type to use')

    # Language model parser
    lm_parser = subparsers.add_parser('lm', help='Language model')
    lm_parser.add_argument('-m', '--model_name', type=str, default='bert-base-uncased', help='Model name to use')
    lm_parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate for training')
    lm_parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    lm_parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')

    llm_parser = subparsers.add_parser('llm', help='Large language model')
    llm_parser.add_argument('-m', '--model_name', type=str, default='gpt-5', help='LLM model name to use')
    llm_parser.add_argument('--max_workers', type=int, default=32, help='Maximum number of workers for parallel prediction')

    return parser.parse_args()

if __name__ == '__main__':
    xgb.set_config(verbosity=1)
    extra_params_dict = {
        'XGB': {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1,
            'n_jobs': -1,
            'device': 'cuda'
        },
        'RF': {
            'n_estimators': 100,
            'max_depth': 10,
            'n_jobs': -1
        },
    }
    args = parse_args()
    pipe = args.pipe
    del args.pipe
    if pipe == 'lm':
        metrics = work_lm(**vars(args))
    elif pipe == 'ml':
        extra_params = extra_params_dict.get(args.model_type, {})
        params = vars(args)
        params.update(extra_params)
        metrics = work_ml(**params)
    elif pipe == 'llm':
        metrics = work_llm(**vars(args))
    else:
        raise NotImplementedError(f"Pipeline type {pipe} is not supported.")
    if args.debug:
        logger.info(f"Final metrics: {metrics}")
