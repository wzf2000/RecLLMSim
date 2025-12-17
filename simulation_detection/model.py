import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class MLModel():
    def __init__(self, type: str, **kwargs):
        self.vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, stop_words=[
            '帮助', '建议', '感谢您', '提供', '想法', '我们', '起来', '我会', '探索', '喜欢',
            '特别', '非常', '高兴', '确保', '选择', '感激', '这些', '你好', '任何', '比较',
            '考虑', '关于', '开始', '一个', '推荐', '指导', '一些', '感谢', '一定', '谢谢',
            '再次', '谢谢您', '真的', '正在', '编辑', '很棒',
        ])
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
        print(f"Training model {self.type} with shape {X_encoded.shape}")
        self.model.fit(X_encoded, y_train)
        if self.type == 'LR':
            self.model: LogisticRegression
            # output the words with top-10 coefficients
            coef = self.model.coef_[0]
            top10_idx = np.argsort(np.abs(coef))[-10:]
            top10_words = [self.vectorizer.get_feature_names_out()[i] for i in top10_idx]
            top10_values = coef[top10_idx]
            print("Top 10 words with highest coefficients:")
            for word, value in zip(top10_words, top10_values):
                print(f"{word}: {value}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_encoded = self.vectorizer.transform(X_test)
        return self.model.predict(X_encoded)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_encoded = self.vectorizer.transform(X_test)
        return self.model.predict_proba(X_encoded)
