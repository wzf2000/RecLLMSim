import warnings
import numpy as np
import xgboost as xgb
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.exceptions import UndefinedMetricWarning

from data_util import ModelType
from evaluate_util import compute_metrics
from pipe_util import exp_sim, exp_sim2human, exp_sim2human2, exp_human, exp_sim4human, exp_sim4human2, exp_human2sim, exp_human2sim2, exp_sim4human3, exp_sim4human4

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

class Model():
    def __init__(self, Type: str, strategy: str = 'OneVsRest', **kwargs):
        self.vectorizer = TfidfVectorizer()
        self.Type = Type
        self.strategy = strategy
        if Type == 'LR':
            model = LogisticRegression(**kwargs)
        elif Type == 'RF':
            model = RandomForestClassifier(**kwargs)
        elif Type == 'XGB':
            model = xgb.XGBClassifier(**kwargs)
        else:
            raise NotImplementedError
        if strategy == 'OneVsRest':
            self.model = OneVsRestClassifier(model)
        elif strategy == 'MultiOutput':
            self.model = MultiOutputClassifier(model)
        elif strategy == 'ClassifierChain':
            self.model = ClassifierChain(model, order='random', random_state=42)
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.Type}-{self.strategy}"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_encoded = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_encoded, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_encoded = self.vectorizer.transform(X_test)
        return self.model.predict(X_encoded)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_encoded = self.vectorizer.transform(X_test)
        if isinstance(self.model, MultiOutputClassifier):
            probs = self.model.predict_proba(X_encoded)
            return np.array([prob[:, 1] if prob.shape[1] == 2 else 1 - prob[:, 0] for prob in probs]).T
        else:
            return self.model.predict_proba(X_encoded)

def work(X_train: list[str], y_train: np.ndarray, X_test: list[str], y_test: np.ndarray, item: str, model_name: str, labels: np.ndarray, ckpt_dir_name: str | None = None, **kwargs) -> dict[str, float]:
    print(item, y_train.shape, y_test.shape)
    Type, strategy = model_name.split('-')
    model = Model(Type, strategy, **kwargs)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    return compute_metrics(y_test, y_score)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, required=True, choices=['sim', 'sim2human', 'human', 'human2sim', 'sim2human2', 'human2sim', 'human2sim2', 'sim4human', 'sim4human2', 'sim4human3', 'sim4human4'])
    parser.add_argument('-l', '--language', type=str, default='zh', choices=['zh', 'en'])
    parser.add_argument('-d', '--data_version', type=int, default=1, choices=[1, 2], help='1: original data; 2: updated data')
    parser.add_argument('-c', '--chat_model', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    xgb.set_config(verbosity=1)
    XGB_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'device': 'cuda:7',
        'tree_method': 'hist',
        'predictor': 'gpu_predictor',
    }
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'n_jobs': -1
    }

    args = parse_args()
    Type, strategy = args.model.split('-')
    if Type == 'XGB':
        params = XGB_PARAMS
    elif Type == 'RF':
        params = RF_PARAMS
    else:
        raise NotImplementedError
    if args.type == 'sim':
        exp_sim(args.model, ModelType.ML, work, args.language, **params)
    elif args.type == 'human':
        exp_human(args.model, ModelType.ML, work, data_version=args.data_version, chat_model=args.chat_model, **params)
    elif args.type == 'sim2human':
        exp_sim2human(args.model, ModelType.ML, work, **params)
    elif args.type == 'human2sim':
        exp_human2sim(args.model, ModelType.ML, work, **params)
    elif args.type == 'sim2human2':
        exp_sim2human2(args.model, ModelType.ML, work, **params)
    elif args.type == 'human2sim':
        exp_human2sim2(args.model, ModelType.ML, work, **params)
    elif args.type == 'human2sim2':
        exp_human2sim2(args.model, ModelType.ML, work, **params)
    elif args.type == 'sim4human':
        exp_sim4human(args.model, ModelType.ML, work, data_version=args.data_version, chat_model=args.chat_model, **params)
    elif args.type == 'sim4human2':
        exp_sim4human2(args.model, ModelType.ML, work, data_version=args.data_version, chat_model=args.chat_model, **params)
    elif args.type == 'sim4human3':
        exp_sim4human3(args.model, ModelType.ML, work, data_version=args.data_version, chat_model=args.chat_model, **params)
    elif args.type == 'sim4human4':
        exp_sim4human4(args.model, ModelType.ML, work, data_version=args.data_version, chat_model=args.chat_model, **params)
    else:
        raise NotImplementedError
