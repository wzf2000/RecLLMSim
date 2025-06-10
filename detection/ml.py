import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from dataset import preprocess_data_ml
from evaluation import evaluate

class MLModel:
    def __init__(self, vectorizer: str = 'tfidf', model_name: str = 'RF', **kwargs):
        if model_name == 'RF':
            self.model = RandomForestClassifier(**kwargs, class_weight='balanced')
        elif model_name == 'RFReg':
            self.model = RandomForestRegressor(**kwargs)
        elif model_name == 'XGB':
            self.model = xgb.XGBClassifier(**kwargs, scale_pos_weight=0.2)
        elif model_name == 'XGBReg':
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            raise NotImplementedError
        if vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        elif vectorizer == 'count':
            self.vectorizer = CountVectorizer()
        else:
            raise NotImplementedError

    def fit(self, X_train: list[str], y_train: np.ndarray):
        X_encoded = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_encoded, y_train)

    def predict(self, X_test: list[str]) -> np.ndarray:
        X_encoded = self.vectorizer.transform(X_test)
        return self.model.predict(X_encoded)

def evaluate_ml(train_data: list[dict], test_data: list[dict], vectorizer: str = 'tfidf', model_name: str = 'RF', profile: bool = False) -> tuple[float, float]:
    X_train, y_train = preprocess_data_ml(train_data, profile=profile)
    X_test, y_test = preprocess_data_ml(test_data, profile=profile)
    model = MLModel(vectorizer=vectorizer, model_name=model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}, Vectorizer: {vectorizer}")
    return evaluate(y_pred, y_test)
