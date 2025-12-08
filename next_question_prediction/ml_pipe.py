import numpy as np
import xgboost as xgb
from loguru import logger
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB


class MLModel:
    def __init__(self, model_type: str, **kwargs):
        if '-' in model_type:
            self.model_type, self.vectorizer_type = model_type.split('-', 1)
        else:
            self.vectorizer_type = 'TFIDF'  # default vectorizer
            self.model_type = model_type
        if self.vectorizer_type == 'TFIDF':
            self.vectorizer = TfidfVectorizer()
        elif self.vectorizer_type == 'Count':
            self.vectorizer = CountVectorizer()
        elif self.vectorizer_type == 'new':
            self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, max_features=5000)
        else:
            raise NotImplementedError(f"Vectorizer type {self.vectorizer_type} is not supported.")
        if self.model_type == 'Random' or self.model_type == 'RandomV2' or self.model_type == 'Pop':
            self.model = None
        elif self.model_type == 'LR':
            self.model = LogisticRegression(**kwargs)
        elif self.model_type == 'RF':
            self.model = RandomForestClassifier(**kwargs)
        elif self.model_type == 'XGB':
            self.model = xgb.XGBClassifier(**kwargs)
        elif self.model_type == 'SVM':
            self.model = SVC(probability=True, **kwargs)
        elif self.model_type == 'GNB':
            self.model = GaussianNB(**kwargs)
        elif self.model_type == 'CNB':
            self.model = ComplementNB(**kwargs)
        elif self.model_type == 'MNB':
            self.model = MultinomialNB(**kwargs)
        else:
            raise NotImplementedError(f"Model model_type {model_type} is not supported.")

    def __str__(self) -> str:
        return f"{self.vectorizer_type}-{self.model_type}"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, aug_samples: int = 0):
        if self.model_type == 'Random':
            logger.info("Using random predictions, no training needed.")
            self.class_num = len(set(y_train))
            return
        elif self.model_type == 'RandomV2':
            logger.info("Using random predictions, no training needed.")
            self.class_num = len(set(y_train))
            counter = Counter(y_train)
            self.class_probs = np.array([counter[i] for i in range(self.class_num)])
            self.class_probs = self.class_probs / self.class_probs.sum()
            return
        elif self.model_type == 'Pop':
            logger.info("Using popularity-based predictions, no training needed.")
            self.class_num = len(set(y_train))
            counter = Counter(y_train)
            self.most_common_class = counter.most_common(1)[0][0]
            class_dist = np.bincount(y_train) / len(y_train)
            class_balance_entropy = -np.sum(class_dist * np.log(class_dist + 1e-12))
            max_entropy = np.log(self.class_num)
            class_balance_entropy /= max_entropy
            logger.info(f"Class distribution: {class_dist}, Entropy: {class_balance_entropy:.4f}")
            return
        if self.vectorizer_type == 'new' and aug_samples > 0:
            # For 'new' vectorizer, we refit on the augmented data only
            self.vectorizer.fit(X_train[aug_samples:])
            X_encoded = self.vectorizer.transform(X_train)
        else:
            X_encoded = self.vectorizer.fit_transform(X_train)
        print(self.vectorizer.get_feature_names_out())
        logger.info(f"Training model {self.model_type} with input shape {X_encoded.shape}")
        if self.model_type in ['GNB', 'CNB', 'MNB']:
            X_encoded = X_encoded.toarray()
        self.model.fit(X_encoded, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model_type == 'Random':
            np.random.seed(42)
            logger.info("Using random predictions.")
            return np.random.choice(self.class_num, size=len(X_test))
        elif self.model_type == 'RandomV2':
            np.random.seed(42)
            logger.info("Using random predictions with class distribution.")
            return np.random.choice(self.class_num, size=len(X_test), p=self.class_probs)
        elif self.model_type == 'Pop':
            logger.info("Using popularity-based predictions.")
            return np.array([self.most_common_class] * len(X_test))
        X_encoded = self.vectorizer.transform(X_test)
        if self.model_type in ['GNB', 'CNB', 'MNB']:
            X_encoded = X_encoded.toarray()
        return self.model.predict(X_encoded)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.model_type == 'Random':
            np.random.seed(42)
            logger.info("Using random predictions.")
            proba = np.random.rand(len(X_test), self.class_num)
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba
        elif self.model_type == 'RandomV2':
            np.random.seed(42)
            logger.info("Using random predictions with class distribution.")
            proba = np.tile(self.class_probs, (len(X_test), 1))
            return proba
        elif self.model_type == 'Pop':
            logger.info("Using popularity-based predictions.")
            proba = np.zeros((len(X_test), self.class_num))
            proba[:, self.most_common_class] = 1.0
            return proba
        X_encoded = self.vectorizer.transform(X_test)
        if self.model_type in ['GNB', 'CNB', 'MNB']:
            X_encoded = X_encoded.toarray()
        return self.model.predict_proba(X_encoded)
