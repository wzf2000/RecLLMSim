import json
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_util import ModelType, get_human_data, get_sim_data
from eval_util import compute_classification_metrics
from model import MLModel
from data_util import SIM_DIR, SIM_DIR_V2, format_history


def work(Type: str, **kwargs) -> dict[str, float]:
    X_human, y_human = get_human_data(model_type=ModelType.ML)
    X_sim, y_sim = get_sim_data(model_type=ModelType.ML, sim_dir=SIM_DIR_V2)
    X = X_human + X_sim
    y = y_human + y_sim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLModel(Type, **kwargs)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)
    metrics = compute_classification_metrics(y_test, y_probs[:, 1])
    sim_probs = model.predict_proba(X_sim)
    human_probs = model.predict_proba(X_human)
    # plot the histogram of the probabilities
    plt.figure(figsize=(10, 6))
    plt.hist([sim_probs[:, 1], human_probs[:, 1]], bins=50, alpha=0.7, label=['Simulated', 'Human'], color=['red', 'blue'])
    plt.title(f'Histogram of Predicted Probabilities ({Type})')
    plt.xlabel('Predicted Probability of Being Simulated')
    plt.legend()
    plt.savefig(f"tmp/probability_histogram_{Type}.png")
    X_sim_rewritten, _ = get_sim_data(model_type=ModelType.ML, sim_dir=SIM_DIR_V2, rewritten=True)
    sim_rewritten_probs = model.predict_proba(X_sim_rewritten)
    accuracy = (sum(sim_rewritten_probs[:, 1] < 0.5) / len(sim_rewritten_probs))
    metrics['rewritten_sim_accuracy'] = accuracy
    # plot the histogram of the probabilities after rewriting
    plt.figure(figsize=(10, 6))
    plt.hist([sim_probs[:, 1], sim_rewritten_probs[:, 1]], bins=50, alpha=0.7, label=['Original Simulated', 'Rewritten Simulated'], color=['red', 'green'])
    plt.title(f'Histogram of Predicted Probabilities After Rewriting ({Type})')
    plt.xlabel('Predicted Probability of Being Simulated')
    plt.legend()
    plt.savefig(f"tmp/probability_histogram_rewritten_{Type}.png")
    return metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation Detection ML Pipeline")
    parser.add_argument('--model_type', type=str, choices=['LR', 'RF', 'XGB'], default='XGB', help='Type of ML model to use')
    return parser.parse_args()

if __name__ == '__main__':
    xgb.set_config(verbosity=1)
    extra_param_dict = {
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
        }
    }
    args = parse_args()
    metrics = work(args.model_type, **extra_param_dict.get(args.model_type, {}))
    print(f"Metrics for model {args.model_type}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
