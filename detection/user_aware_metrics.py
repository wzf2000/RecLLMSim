"""
用户感知评估指标 (User-Aware Metrics)

解决问题：不同用户对满意度的打分存在系统性偏差（基准分不同），
直接计算全局 Pearson/Spearman/Kappa 会混入用户间差异，
导致指标虚高（用户间差异被误算为预测能力）。

两种去偏策略：
─────────────────────────────────────────────────────────────────────
A. Per-user aggregation（用户内独立计算，再汇总）
   - 对每个用户单独计算 Pearson / Spearman / MAE / Kappa
   - 按样本数加权平均（相关系数先做 Fisher z-transform 再平均）
   - 过滤样本数不足 min_samples 的用户（相关系数不可靠）
   - 优点：直接衡量"模型在单个用户内排序是否正确"
   - 缺点：样本少的用户结果不稳定

B. Within-user mean-centering（去均值后全局计算）
   - 对每个用户，gold / pred 分别减去该用户均值
   - 在所有残差上计算全局 Pearson / Spearman / MAE
   - 不能用于 Kappa（中心化后分数不再是整数类别）
   - 优点：利用全部数据，消除用户基准差异
   - 缺点：消除了用户间绝对量级信息
─────────────────────────────────────────────────────────────────────
"""

import math
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────────────────

def _fisher_z(r: float) -> float:
    """Pearson/Spearman r → Fisher z-score（atanh 变换）。"""
    r = max(-0.9999, min(0.9999, r))  # 避免 atanh(±1)
    return math.atanh(r)


def _inv_fisher_z(z: float) -> float:
    return math.tanh(z)


def _weighted_mean_r(rs: list[float], ns: list[int]) -> float:
    """用 Fisher z-transform 做加权平均相关系数。"""
    if not rs:
        return float("nan")
    zs = [_fisher_z(r) for r in rs]
    total_n = sum(ns)
    if total_n == 0:
        return float("nan")
    z_mean = sum(z * n for z, n in zip(zs, ns)) / total_n
    return _inv_fisher_z(z_mean)


def _group_by_user(
    gold: list[float],
    pred: list[float],
    users: list[str],
) -> dict[str, tuple[list[float], list[float]]]:
    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for g, p, u in zip(gold, pred, users):
        groups[u][0].append(g)
        groups[u][1].append(p)
    return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# 策略 A：Per-user aggregation
# ──────────────────────────────────────────────────────────────────────────────

def per_user_aggregated(
    gold: list[float],
    pred: list[float],
    users: list[str],
    min_samples: int = 3,
) -> dict[str, float]:
    """
    每个用户独立计算指标，再按样本数加权汇总。

    返回字段（前缀 pu_ = per-user aggregated）：
      pu_pearson, pu_spearman, pu_mae, pu_rmse, pu_kappa
      pu_n_users_total, pu_n_users_used, pu_n_users_skipped
      pu_min_samples_threshold
    """
    groups = _group_by_user(gold, pred, users)

    pearson_list, spearman_list, kappa_list = [], [], []
    mae_list, rmse_list = [], []
    ns_corr: list[int] = []   # 用于相关系数加权（仅满足 min_samples 的用户）
    ns_kappa: list[int] = []
    ns_mae: list[int] = []

    skipped_corr = 0
    skipped_kappa = 0

    for user, (g_list, p_list) in groups.items():
        n = len(g_list)
        g_arr = np.array(g_list, dtype=float)
        p_arr = np.array(p_list, dtype=float)

        # MAE / RMSE：所有用户都算（哪怕只有 1 条）
        mae_list.append(mean_absolute_error(g_arr, p_arr))
        rmse_list.append(root_mean_squared_error(g_arr, p_arr))
        ns_mae.append(n)

        # Pearson / Spearman：需要 min_samples 且有方差
        if n < min_samples:
            skipped_corr += 1
        else:
            g_std = g_arr.std()
            p_std = p_arr.std()
            if g_std < 1e-9 or p_std < 1e-9:
                # 某一侧方差为 0，相关系数无意义，跳过
                skipped_corr += 1
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r_p = pearsonr(g_arr, p_arr)[0]
                    r_s = spearmanr(g_arr, p_arr)[0]
                if not math.isnan(r_p):
                    pearson_list.append(r_p)
                    spearman_list.append(r_s if not math.isnan(r_s) else 0.0)
                    ns_corr.append(n)

        # Kappa：需要 min_samples 且至少 2 个不同分数值
        if n < min_samples:
            skipped_kappa += 1
        else:
            g_int = np.round(g_arr).astype(int).tolist()
            p_int = np.round(p_arr).astype(int).tolist()
            if len(set(g_int)) < 2:
                # 该用户所有 gold 相同 → kappa 无意义
                skipped_kappa += 1
            else:
                try:
                    labels = list(range(1, 6))
                    k = cohen_kappa_score(g_int, p_int, weights="quadratic", labels=labels)
                    kappa_list.append(k)
                    ns_kappa.append(n)
                except Exception:
                    skipped_kappa += 1

    n_users = len(groups)
    n_used_corr = len(pearson_list)

    result = {
        "pu_pearson":  _weighted_mean_r(pearson_list, ns_corr),
        "pu_spearman": _weighted_mean_r(spearman_list, ns_corr),
        "pu_mae":   float(np.average(mae_list, weights=ns_mae)) if mae_list else float("nan"),
        "pu_rmse":  float(np.average(rmse_list, weights=ns_mae)) if rmse_list else float("nan"),
        "pu_kappa": (
            float(np.average(kappa_list, weights=ns_kappa)) if kappa_list else float("nan")
        ),
        "pu_n_users_total":   n_users,
        "pu_n_users_used_corr": n_used_corr,
        "pu_n_users_skipped_corr": skipped_corr,
        "pu_n_users_used_kappa": len(kappa_list),
        "pu_n_users_skipped_kappa": skipped_kappa,
        "pu_min_samples_threshold": min_samples,
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 策略 B：Within-user mean-centering
# ──────────────────────────────────────────────────────────────────────────────

def within_user_centered(
    gold: list[float],
    pred: list[float],
    users: list[str],
) -> dict[str, float]:
    """
    对每个用户分别对 gold/pred 做 mean-centering，
    再在全部残差上计算全局指标。

    返回字段（前缀 wc_ = within-user centered）：
      wc_pearson, wc_spearman, wc_mae, wc_rmse
    """
    groups = _group_by_user(gold, pred, users)

    g_centered_all: list[float] = []
    p_centered_all: list[float] = []

    for user, (g_list, p_list) in groups.items():
        g_mean = float(np.mean(g_list))
        p_mean = float(np.mean(p_list))
        g_centered_all.extend([g - g_mean for g in g_list])
        p_centered_all.extend([p - p_mean for p in p_list])

    g_arr = np.array(g_centered_all, dtype=float)
    p_arr = np.array(p_centered_all, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_p = pearsonr(g_arr, p_arr)[0] if g_arr.std() > 1e-9 else float("nan")
        r_s = spearmanr(g_arr, p_arr)[0] if p_arr.std() > 1e-9 else float("nan")

    return {
        "wc_pearson":  float(r_p),
        "wc_spearman": float(r_s),
        "wc_mae":  float(mean_absolute_error(g_arr, p_arr)),
        "wc_rmse": float(root_mean_squared_error(g_arr, p_arr)),
        "wc_n_users": len(groups),
        "wc_n_samples": len(g_centered_all),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 组合入口
# ──────────────────────────────────────────────────────────────────────────────

def compute_user_aware_metrics(
    gold: list[float],
    pred: list[float],
    users: list[str],
    min_samples: int = 3,
) -> dict[str, float]:
    """
    同时计算 per-user aggregation 和 within-user centering 两套指标并合并返回。
    """
    pu = per_user_aggregated(gold, pred, users, min_samples=min_samples)
    wc = within_user_centered(gold, pred, users)
    return {**pu, **wc}


# ──────────────────────────────────────────────────────────────────────────────
# 打印工具
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v: float, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if not math.isnan(v) else "  N/A  "


def print_user_aware_metrics(m: dict[str, float], header: str = "") -> None:
    sep = "=" * 62
    from loguru import logger
    if header:
        logger.info(sep)
        logger.info(f"  {header}")
    logger.info(sep)
    logger.info("  ── 策略 A：Per-user aggregation ──────────────────────────")
    n_tot = int(m.get("pu_n_users_total", 0))
    n_corr = int(m.get("pu_n_users_used_corr", 0))
    n_skip_c = int(m.get("pu_n_users_skipped_corr", 0))
    n_kappa = int(m.get("pu_n_users_used_kappa", 0))
    n_skip_k = int(m.get("pu_n_users_skipped_kappa", 0))
    thresh = int(m.get("pu_min_samples_threshold", 3))
    logger.info(f"  用户总数:      {n_tot}  (相关系数: {n_corr} 用户 / 跳过 {n_skip_c},"
                f"  Kappa: {n_kappa} / 跳过 {n_skip_k},  阈值 min_samples={thresh})")
    logger.info(f"  Pearson  (pu): {_fmt(m['pu_pearson'])}"
                f"    Spearman (pu): {_fmt(m['pu_spearman'])}")
    logger.info(f"  Kappa    (pu): {_fmt(m['pu_kappa'])}"
                f"    MAE      (pu): {_fmt(m['pu_mae'])}"
                f"    RMSE (pu): {_fmt(m['pu_rmse'])}")
    logger.info("  ── 策略 B：Within-user mean-centering ────────────────────")
    logger.info(f"  用户数: {int(m.get('wc_n_users', 0))}  样本数: {int(m.get('wc_n_samples', 0))}")
    logger.info(f"  Pearson  (wc): {_fmt(m['wc_pearson'])}"
                f"    Spearman (wc): {_fmt(m['wc_spearman'])}")
    logger.info(f"  MAE      (wc): {_fmt(m['wc_mae'])}"
                f"    RMSE     (wc): {_fmt(m['wc_rmse'])}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 二分类（SAT/DSAT）用户感知指标
#
# 与 1-5 分制版本思路相同，但指标换为 accuracy / F1 / precision / recall /
# kappa / AUC 等二分类指标。
# 策略 A（per-user aggregation）：逐用户计算后加权平均。
# 策略 B（within-user centering）：对 gold (0/1) 和 confidence 各减去
#   用户均值，在全部残差上计算 Pearson / Spearman / MAE / RMSE，
#   衡量模型在单用户内部的区分能力。
# ──────────────────────────────────────────────────────────────────────────────

def _group_by_user_binary(
    gold: list[int],
    pred: list[int],
    users: list[str],
    confidence: list[float] | None = None,
) -> dict[str, dict[str, list]]:
    """按用户分组，每组含 gold(int), pred(int), conf(float)。"""
    has_conf = confidence is not None
    groups: dict[str, dict[str, list]] = defaultdict(
        lambda: {"gold": [], "pred": [], "conf": []}
    )
    for i, (g, p, u) in enumerate(zip(gold, pred, users)):
        groups[u]["gold"].append(g)
        groups[u]["pred"].append(p)
        if has_conf:
            groups[u]["conf"].append(confidence[i])
    return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# 策略 A：Per-user aggregation（二分类版）
# ──────────────────────────────────────────────────────────────────────────────

def per_user_aggregated_binary(
    gold: list[int],
    pred: list[int],
    users: list[str],
    confidence: list[float] | None = None,
    min_samples: int = 3,
) -> dict[str, float]:
    """
    二分类版 per-user aggregation。

    每个用户独立计算 accuracy / F1 / precision / recall / kappa / AUC，
    再按样本数加权平均。
    跳过条件：样本数 < min_samples，或 gold 只有单一类别。
    """
    groups = _group_by_user_binary(gold, pred, users, confidence)

    acc_l, f1m_l, f1s_l, f1d_l = [], [], [], []
    prec_l, rec_l = [], []
    kappa_l, auc_l = [], []
    ns_cls: list[int] = []
    ns_kappa: list[int] = []
    ns_auc: list[int] = []

    skipped = 0
    kappa_extra_skip = 0
    auc_extra_skip = 0

    for _user, data in groups.items():
        g, p, c = data["gold"], data["pred"], data["conf"]
        n = len(g)

        if n < min_samples or len(set(g)) < 2:
            skipped += 1
            continue

        acc_l.append(accuracy_score(g, p))
        f1m_l.append(f1_score(g, p, average="macro", zero_division=0))
        f1s_l.append(f1_score(g, p, pos_label=1, average="binary", zero_division=0))
        f1d_l.append(f1_score(g, p, pos_label=0, average="binary", zero_division=0))
        prec_l.append(precision_score(g, p, pos_label=1, average="binary", zero_division=0))
        rec_l.append(recall_score(g, p, pos_label=1, average="binary", zero_division=0))
        ns_cls.append(n)

        try:
            kappa_l.append(cohen_kappa_score(g, p))
            ns_kappa.append(n)
        except Exception:
            kappa_extra_skip += 1

        if c:
            try:
                auc_l.append(roc_auc_score(g, c))
                ns_auc.append(n)
            except Exception:
                auc_extra_skip += 1
        else:
            auc_extra_skip += 1

    def _wavg(vals: list[float], ws: list[int]) -> float:
        return float(np.average(vals, weights=ws)) if vals else float("nan")

    n_users = len(groups)
    n_used = len(acc_l)

    return {
        "pu_bin_accuracy":      _wavg(acc_l, ns_cls),
        "pu_bin_f1_macro":      _wavg(f1m_l, ns_cls),
        "pu_bin_f1_sat":        _wavg(f1s_l, ns_cls),
        "pu_bin_f1_dsat":       _wavg(f1d_l, ns_cls),
        "pu_bin_precision_sat": _wavg(prec_l, ns_cls),
        "pu_bin_recall_sat":    _wavg(rec_l, ns_cls),
        "pu_bin_kappa":         _wavg(kappa_l, ns_kappa),
        "pu_bin_auc":           _wavg(auc_l, ns_auc),
        "pu_bin_n_users_total":       n_users,
        "pu_bin_n_users_used":        n_used,
        "pu_bin_n_users_skipped":     skipped,
        "pu_bin_n_users_used_kappa":  len(kappa_l),
        "pu_bin_n_users_skipped_kappa": skipped + kappa_extra_skip,
        "pu_bin_n_users_used_auc":    len(auc_l),
        "pu_bin_n_users_skipped_auc": skipped + auc_extra_skip,
        "pu_bin_min_samples_threshold": min_samples,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 策略 B：Within-user centering（二分类版）
# ──────────────────────────────────────────────────────────────────────────────

def within_user_centered_binary(
    gold: list[int],
    pred: list[int],
    users: list[str],
    confidence: list[float] | None = None,
) -> dict[str, float]:
    """
    二分类版 within-user centering。

    gold (0/1) 和 score（优先用 confidence，否则用 pred (0/1)）
    各减去用户均值后，在全部残差上计算 Pearson / Spearman / MAE / RMSE。

    离散指标（accuracy, F1, kappa）不适用于中心化后的连续值，
    因此本策略仅返回相关系数和回归类指标。
    """
    groups = _group_by_user_binary(gold, pred, users, confidence)

    g_centered: list[float] = []
    s_centered: list[float] = []

    for _user, data in groups.items():
        g = data["gold"]
        s = data["conf"] if data["conf"] else [float(x) for x in data["pred"]]
        g_mean = float(np.mean(g))
        s_mean = float(np.mean(s))
        g_centered.extend(gi - g_mean for gi in g)
        s_centered.extend(si - s_mean for si in s)

    g_arr = np.array(g_centered, dtype=float)
    s_arr = np.array(s_centered, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        can_corr = g_arr.std() > 1e-9 and s_arr.std() > 1e-9
        r_p = float(pearsonr(g_arr, s_arr)[0]) if can_corr else float("nan")
        r_s = float(spearmanr(g_arr, s_arr)[0]) if can_corr else float("nan")

    return {
        "wc_bin_pearson":   r_p,
        "wc_bin_spearman":  r_s,
        "wc_bin_mae":  float(mean_absolute_error(g_arr, s_arr)),
        "wc_bin_rmse": float(root_mean_squared_error(g_arr, s_arr)),
        "wc_bin_n_users":   len(groups),
        "wc_bin_n_samples": len(g_centered),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 组合入口（二分类版）
# ──────────────────────────────────────────────────────────────────────────────

def compute_user_aware_binary_metrics(
    gold: list[int],
    pred: list[int],
    users: list[str],
    confidence: list[float] | None = None,
    min_samples: int = 3,
) -> dict[str, float]:
    """同时计算二分类的 per-user aggregation 和 within-user centering。"""
    pu = per_user_aggregated_binary(gold, pred, users, confidence, min_samples)
    wc = within_user_centered_binary(gold, pred, users, confidence)
    return {**pu, **wc}


# ──────────────────────────────────────────────────────────────────────────────
# 打印工具（二分类版）
# ──────────────────────────────────────────────────────────────────────────────

def print_user_aware_binary_metrics(m: dict[str, float], header: str = "") -> None:
    sep = "=" * 62
    from loguru import logger
    if header:
        logger.info(sep)
        logger.info(f"  {header}")
    logger.info(sep)
    logger.info("  ── 策略 A：Per-user aggregation（二分类）─────────────────")
    n_tot  = int(m.get("pu_bin_n_users_total", 0))
    n_used = int(m.get("pu_bin_n_users_used", 0))
    n_skip = int(m.get("pu_bin_n_users_skipped", 0))
    n_kap  = int(m.get("pu_bin_n_users_used_kappa", 0))
    n_auc  = int(m.get("pu_bin_n_users_used_auc", 0))
    thresh = int(m.get("pu_bin_min_samples_threshold", 3))
    logger.info(f"  用户总数: {n_tot}  (分类: {n_used} / 跳过 {n_skip},"
                f"  Kappa: {n_kap},  AUC: {n_auc},  阈值 min_samples={thresh})")
    logger.info(f"  Accuracy (pu): {_fmt(m['pu_bin_accuracy'])}"
                f"    F1-macro (pu): {_fmt(m['pu_bin_f1_macro'])}")
    logger.info(f"  F1-SAT   (pu): {_fmt(m['pu_bin_f1_sat'])}"
                f"    F1-DSAT  (pu): {_fmt(m['pu_bin_f1_dsat'])}")
    logger.info(f"  Prec-SAT (pu): {_fmt(m['pu_bin_precision_sat'])}"
                f"    Rec-SAT  (pu): {_fmt(m['pu_bin_recall_sat'])}")
    logger.info(f"  Kappa    (pu): {_fmt(m['pu_bin_kappa'])}"
                f"    AUC      (pu): {_fmt(m['pu_bin_auc'])}")
    logger.info("  ── 策略 B：Within-user centering（二分类）───────────────")
    logger.info(f"  用户数: {int(m.get('wc_bin_n_users', 0))}"
                f"  样本数: {int(m.get('wc_bin_n_samples', 0))}")
    logger.info(f"  Pearson  (wc): {_fmt(m['wc_bin_pearson'])}"
                f"    Spearman (wc): {_fmt(m['wc_bin_spearman'])}")
    logger.info(f"  MAE      (wc): {_fmt(m['wc_bin_mae'])}"
                f"    RMSE     (wc): {_fmt(m['wc_bin_rmse'])}")
    logger.info(sep)
