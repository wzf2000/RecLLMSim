import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from log_util import add_log
from data_util import ModelType, get_human_intent_data
from pipe_util import ExpType, split_train_test
from evaluate_util import compute_classification_metrics

class Model():
    def __init__(self, type: str, **kwargs):
        self.vectorizer = TfidfVectorizer()
        self.type = type
        if type == 'LR':
            self.model = LogisticRegression(**kwargs)
        elif type == 'RF':
            self.model = RandomForestClassifier(**kwargs)
        elif type == 'XGB':
            self.model = xgb.XGBClassifier(**kwargs)
        else:
            raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{self.type}"
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_encoded = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_encoded, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_encoded = self.vectorizer.transform(X_test)
        return self.model.predict(X_encoded)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_encoded = self.vectorizer.transform(X_test)
        return self.model.predict_proba(X_encoded)

def work_human(Type: str, **kwargs) -> dict[str, float]:
    X, y = get_human_intent_data(model_type=ModelType.ML)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = Model(Type, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)
    add_log('intent', str(model), ExpType.HUMAN, metrics, cls=True)
    return metrics

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
    # work_human('LR')
    # work_human('RF', **RF_PARAMS)
    work_human('XGB', **XGB_PARAMS)
