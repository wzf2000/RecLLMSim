import warnings
import numpy as np
import xgboost as xgb
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
    def __init__(self, type: str, strategy: str = 'OneVsRest', **kwargs):
        self.vectorizer = TfidfVectorizer()
        self.type = type
        self.strategy = strategy
        if type == 'LR':
            model = LogisticRegression(**kwargs)
        elif type == 'RF':
            model = RandomForestClassifier(**kwargs)
        elif type == 'XGB':
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
        return f"{self.type}-{self.strategy}"

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

def work(X_train: list[str], y_train: np.ndarray, X_test: list[str], y_test: np.ndarray, item: str, model_name: str, labels: np.ndarray, **kwargs) -> dict[str, float]:
    Type, strategy = model_name.split('-')
    model = Model(Type, strategy, **kwargs)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    return compute_metrics(y_test, y_score)

if __name__ == '__main__':
    xgb.set_config(verbosity=1)
    XGB_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'device': 'cuda'
    }
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'n_jobs': -1
    }

    # exp_sim('XGB-MultiOutput', ModelType.ML, work, 'zh', **XGB_PARAMS)
    # exp_sim('XGB-ClassifierChain', ModelType.ML, work, 'zh', **XGB_PARAMS)
    # exp_sim('RF-MultiOutput', ModelType.ML, work, 'zh', **RF_PARAMS)
    # exp_sim('RF-ClassifierChain', ModelType.ML, work, 'zh', **RF_PARAMS)

    # exp_sim2human('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim2human('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim2human('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_sim2human('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)

    # exp_human('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_human('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_human('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_human('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)

    # exp_sim4human('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim4human('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim4human('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_sim4human('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)

    # exp_human2sim('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_human2sim('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_human2sim('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_human2sim('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)

    # exp_sim2human2('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim2human2('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim2human2('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_sim2human2('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)

    # exp_sim4human2('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim4human2('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim4human2('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_sim4human2('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)

    # exp_human2sim2('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_human2sim2('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_human2sim2('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_human2sim2('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)

    # exp_sim4human3('XGB-MultiOutput', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim4human3('XGB-ClassifierChain', ModelType.ML, work, **XGB_PARAMS)
    # exp_sim4human3('RF-MultiOutput', ModelType.ML, work, **RF_PARAMS)
    # exp_sim4human3('RF-ClassifierChain', ModelType.ML, work, **RF_PARAMS)
