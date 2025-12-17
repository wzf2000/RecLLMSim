import os
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from argparse import ArgumentParser, Namespace
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from data_util import ModelType, get_sim_data, get_human_data
from pipe_util import set_seed
from predict_ml import Model, compute_metrics


def train(X_train: list[str], y_train: np.ndarray, model_name: str, **kwargs) -> Model:
    Type, strategy = model_name.split('-')
    model = Model(Type, strategy, **kwargs)
    model.fit(X_train, y_train)
    return model

def get_score(model: Model, X_test: list[str], y_test: np.ndarray) -> None:
    test_score = model.predict_proba(X_test)
    scores = []
    for i in range(len(y_test)):
        gt_labels = np.where(y_test[i] == 1)[0]
        gt_scores = test_score[i][gt_labels]
        gt_score_mean = np.mean(gt_scores)
        scores.append(gt_score_mean)
    return scores

def plot_histogram(item: str, model_name: str, scores_dict: dict[str, list[float]]) -> None:
    labels = list(scores_dict.keys())
    scores_list = [scores_dict[label] for label in labels]
    # Plot histogram
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    os.makedirs(f'figures/simulation-analysis/{item}', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(scores_list, bins=20, alpha=0.7, label=labels, color=colors[:len(labels)])
    plt.title(f'Histogram of Ground Truth Label Scores for {item} using {model_name}')
    plt.xlabel('Mean of Predicted Probabilities for Ground Truth Labels')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.savefig(f'figures/simulation-analysis/{item}/{model_name}_histogram.png')

def pipe(item: str, model_name: str, model_type: ModelType, task: str | None = None, **kwargs) -> None:
    set_seed(42)
    X_sim_1, y_sim_1 = get_sim_data(item, 'zh', task, model_type, version=1, filtered=True, only='user')
    X_sim_2, y_sim_2 = get_sim_data(item, 'zh', task, model_type, version=2, filtered=True, only='user')
    X_sim_3, y_sim_3 = get_sim_data(item, 'zh', task, model_type, version=3, filtered=True, only='user')
    X_sim_4, y_sim_4 = get_sim_data(item, 'zh', task, model_type, version=4, filtered=True, only='user')
    X_sim_5, y_sim_5 = get_sim_data(item, 'zh', task, model_type, version=5, filtered=True, only='user')
    X_human, y_human = get_human_data(item, task, model_type, version=2, only='user')
    y = y_human + y_sim_1 + y_sim_2 + y_sim_3 + y_sim_4 + y_sim_5
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    human_size = len(X_human)
    y_human = y[:human_size]
    y_sim_1 = y[human_size:human_size + len(X_sim_1)]
    y_sim_2 = y[human_size + len(X_sim_1):human_size + len(X_sim_1) + len(X_sim_2)]
    y_sim_3 = y[human_size + len(X_sim_1) + len(X_sim_2):human_size + len(X_sim_1) + len(X_sim_2) + len(X_sim_3)]
    y_sim_4 = y[human_size + len(X_sim_1) + len(X_sim_2) + len(X_sim_3):human_size + len(X_sim_1) + len(X_sim_2) + len(X_sim_3) + len(X_sim_4)]
    y_sim_5 = y[human_size + len(X_sim_1) + len(X_sim_2) + len(X_sim_3) + len(X_sim_4):]
    X_train, X_test, y_train, y_test = train_test_split(X_human, y_human, test_size=0.2, random_state=42)
    logger.info(f"Training model {model_name} for item {item} with {len(X_train)} training samples")
    model = train(X_train, y_train, model_name, **kwargs)
    original_metrics = compute_metrics(y_test, model.predict_proba(X_test), more=True)

    def log_metric(metrics: dict[str, float], prefix: str) -> None:
        log_str = f'{prefix} metrics for item {item} with model {model_name}:\n'
        log_str += '\n'.join([f"{prefix} {k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)

    log_metric(original_metrics, 'Original')
    scores_dict = {
        'Human': get_score(model, X_test, y_test),
        'Simulated_V1': get_score(model, X_sim_1, y_sim_1),
        'Simulated_V2': get_score(model, X_sim_2, y_sim_2),
        'Simulated_V3': get_score(model, X_sim_3, y_sim_3),
        'Simulated_V4': get_score(model, X_sim_4, y_sim_4),
        'Simulated_V5': get_score(model, X_sim_5, y_sim_5),
    }
    plot_histogram(
        item,
        model_name,
        scores_dict
    )
    # filter the simulation data with average score more than 0.3
    filtered_X_sim = []
    filtered_y_sim = []
    sim_data = [
        (X_sim_1, y_sim_1),
        (X_sim_2, y_sim_2),
        (X_sim_3, y_sim_3),
        (X_sim_4, y_sim_4),
        (X_sim_5, y_sim_5),
    ]
    for version, (X_sim, y_sim) in enumerate(sim_data, start=1):
        scores = scores_dict[f'Simulated_V{version}']
        for i in range(len(X_sim)):
            if scores[i] > 0.3:
                filtered_X_sim.append(X_sim[i])
                filtered_y_sim.append(y_sim[i])
    logger.info(f"Filtered simulation data size for item {item} with model {model_name}: {len(filtered_X_sim)}")
    filtered_X_sim = np.array(filtered_X_sim)
    filtered_y_sim = np.array(filtered_y_sim)
    new_X_train = np.concatenate([X_train, filtered_X_sim], axis=0)
    new_y_train = np.concatenate([y_train, filtered_y_sim], axis=0)
    logger.info(f"New training data size for item {item} with model {model_name}: {len(X_train)} + {len(filtered_X_sim)} = {len(new_X_train)}")
    new_model = train(new_X_train, new_y_train, model_name, **kwargs)
    y_probs = new_model.predict_proba(X_test)
    new_metrics = compute_metrics(y_test, y_probs, more=True)
    from sklearn import metrics
    for i, cls in enumerate(mlb.classes_):
        logger.info(f"Tag: {cls}, roc_auc = {metrics.roc_auc_score(y_test[:, i], y_probs[:, i]):.4f}, ratio = {np.sum(y_test[:, i])} / {len(y_test)} = {np.sum(y_test[:, i]) / len(y_test):.4f}")
    log_metric(new_metrics, 'Filtered Simulation Augmented')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-i', '--item', type=str, required=True, help='Profile item to analyze')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Model name in the format `Type-Strategy`')
    parser.add_argument('-t', '--task', type=str, default=None, help='Specific task to filter data')
    return parser.parse_args()

def main():
    args = parse_args()
    pipe(args.item, args.model_name, ModelType.LM, task=args.task)

if __name__ == '__main__':
    main()
