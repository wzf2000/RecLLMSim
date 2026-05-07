from __future__ import annotations

import os

import numpy as np
import tiktoken
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.llm import client


_MAX_EMBEDDING_TOKENS = 8191


def _get_tiktoken_enc(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _truncate_text(text: str, enc: tiktoken.Encoding, max_tokens: int = _MAX_EMBEDDING_TOKENS) -> str:
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[-max_tokens:])


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    """调用 OpenAI Embeddings API，超长文本自动截断至 token 上限。"""
    enc = _get_tiktoken_enc(model)
    texts = [_truncate_text(t, enc) for t in texts]
    resp = client.embeddings.create(input=texts, model=model)
    items = sorted(resp.data, key=lambda x: x.index)
    return [item.embedding for item in items]


def get_embeddings(
    texts: list[str],
    model: str = "text-embedding-ada-002",
    cache_file: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    """
    批量获取文本 embedding，返回 (N, D) numpy 数组。
    cache_file 若指定则自动读写缓存（.npz 格式，同时保存文本指纹用于校验）。
    """
    if cache_file and os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        cached_emb: np.ndarray = data["embeddings"]
        cached_n = int(data["n"])
        if cached_n == len(texts):
            logger.info(f"[Phase 4] 加载 embedding 缓存: {cache_file}  shape={cached_emb.shape}")
            return cached_emb
        logger.info(f"[Phase 4] embedding 缓存大小不匹配 ({cached_n} vs {len(texts)})，重新提取")

    logger.info(f"[Phase 4] 提取 {len(texts)} 条文本的 embedding（model={model}）...")
    enc = _get_tiktoken_enc(model)
    n_truncated = sum(1 for t in texts if len(enc.encode(t)) > _MAX_EMBEDDING_TOKENS)
    if n_truncated:
        logger.warning(
            f"[Phase 4] {n_truncated}/{len(texts)} 条文本超过 {_MAX_EMBEDDING_TOKENS} tokens，将自动截断"
        )
    all_embeddings: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        embs = _embed_batch(batch, model)
        all_embeddings.extend(embs)

    arr = np.array(all_embeddings, dtype=np.float32)
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        np.savez_compressed(cache_file, embeddings=arr, n=len(texts))
        logger.info(f"[Phase 4] embedding 已保存: {cache_file}  shape={arr.shape}")
    return arr


def build_rubric_feature_vec(
    scored_results: list[dict],
    k: int,
) -> np.ndarray:
    """
    将 rubric 评分结果转为二值特征向量。
    每条样本输出长度为 2*k 的向量：
      前 k 维 = SAT rubric match (1/0)
      后 k 维 = DSAT rubric match (1/0)
    rubric 编号从 1 开始（与 LLM 输出一致）。
    """
    n = len(scored_results)
    feats = np.zeros((n, 2 * k), dtype=np.float32)
    for i, r in enumerate(scored_results):
        for idx in r.get("sat_matches", []):
            j = int(idx) - 1
            if 0 <= j < k:
                feats[i, j] = 1.0
        for idx in r.get("dsat_matches", []):
            j = int(idx) - 1
            if 0 <= j < k:
                feats[i, k + j] = 1.0
    return feats


def train_and_eval_classifier(
    train_rubric_feats: np.ndarray,
    train_emb: np.ndarray | None,
    train_labels: list[int],
    test_rubric_feats: np.ndarray,
    test_emb: np.ndarray | None,
    test_labels: list[int],
    test_scored: list[dict],
) -> dict[str, dict]:
    """
    训练 LogisticRegression 分类器并评估，支持以下三种特征组合：
      - rubric_only:    仅用 rubric 二值特征
      - embedding_only: 仅用 text embedding
      - combined:       rubric + embedding 拼接（论文默认）
    返回 {variant_name: metrics_dict}。
    """
    def _build(rubric_f: np.ndarray, emb: np.ndarray | None, use_rubric: bool, use_emb: bool) -> np.ndarray:
        parts = []
        if use_rubric:
            parts.append(rubric_f)
        if use_emb and emb is not None:
            parts.append(emb)
        return np.concatenate(parts, axis=1) if parts else rubric_f

    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "rubric_only": (
            _build(train_rubric_feats, None, True, False),
            _build(test_rubric_feats, None, True, False),
        ),
    }
    if train_emb is not None:
        variants["embedding_only"] = (
            _build(train_rubric_feats, train_emb, False, True),
            _build(test_rubric_feats, test_emb, False, True),
        )
        variants["combined"] = (
            _build(train_rubric_feats, train_emb, True, True),
            _build(test_rubric_feats, test_emb, True, True),
        )

    all_metrics: dict[str, dict] = {}
    for name, (X_train, X_test) in variants.items():
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        clf.fit(X_train_s, train_labels)
        pred = clf.predict(X_test_s).tolist()
        prob = clf.predict_proba(X_test_s)[:, 1].tolist()  # P(SAT)

        try:
            auc = roc_auc_score(test_labels, prob)
        except Exception:
            auc = float("nan")

        m = {
            "accuracy": accuracy_score(test_labels, pred),
            "f1_macro": f1_score(test_labels, pred, average="macro", zero_division=0),
            "f1_sat": f1_score(test_labels, pred, pos_label=1, average="binary", zero_division=0),
            "f1_dsat": f1_score(test_labels, pred, pos_label=0, average="binary", zero_division=0),
            "precision_sat": precision_score(test_labels, pred, pos_label=1, average="binary", zero_division=0),
            "recall_sat": recall_score(test_labels, pred, pos_label=1, average="binary", zero_division=0),
            "kappa": cohen_kappa_score(test_labels, pred),
            "auc": auc,
            "parse_rate": 1.0,
            "n_samples": len(test_labels),
            "n_sat_gold": int(sum(test_labels)),
            "n_dsat_gold": int(len(test_labels) - sum(test_labels)),
        }
        all_metrics[name] = m

    return all_metrics
