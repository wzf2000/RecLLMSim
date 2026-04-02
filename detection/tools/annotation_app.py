import os
import json
import random
import streamlit as st
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)
from scipy.stats import spearmanr, pearsonr

# 假设你的获取数据函数保存在 metric_statistics.py 中
from lib.metric_statistics import get_satisfaction_data
from lib.data_split import split_by_user_group_shuffle_split

# ================= 数据准备模块 =================

@st.cache_data
def load_and_sample_test_data(sample_size: int = 100, ref_per_score: int = 3):
    """
    加载数据，进行与模型训练相同的切分，并从测试集中随机采样用于人工标注
    """
    data_list = get_satisfaction_data()

    # 将原始数据拆解为结构化的单轮评估样本
    structured_samples = []
    for sample in data_list:
        history_turns = []
        assistant_turn_idx = 0
        for utt in sample['history']:
            history_turns.append(
                {"role": utt['role'], "content": utt['content']})
            if utt['role'] == 'assistant':
                structured_samples.append({
                    "profile": sample['profile'],
                    "task_context": sample['task_context'],
                    "history": list(history_turns),  # 截取到当前轮次的历史
                    "gt_score": sample['satisfaction_scores'][assistant_turn_idx],
                    "gt_reason": sample['dissatisfaction_reasons'][assistant_turn_idx],
                    "chat_model": sample['chat_model'],
                    "task": sample['task'],
                    "user": sample.get("user", "unknown"),
                })
                assistant_turn_idx += 1

    # 保持与你的训练脚本完全一致的拆分逻辑，以确保拿到同样的测试集
    train_idx, _, test_idx = split_by_user_group_shuffle_split(
        [s["user"] for s in structured_samples],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    train_samples = [structured_samples[i] for i in train_idx]
    test_samples = [structured_samples[i] for i in test_idx]

    # 从训练集中按分数分层抽取参考样本（用于对照/校准标注尺度）
    random.seed(42)
    ref_by_score: dict[int, list[dict]] = {i: [] for i in range(1, 6)}
    for score in range(1, 6):
        candidates = [s for s in train_samples if int(s["gt_score"]) == score]
        if candidates:
            k = min(ref_per_score, len(candidates))
            ref_by_score[score] = random.sample(candidates, k)

    # 从测试集中随机采样指定数量的数据用于人工标注
    random.seed(42)  # 固定采样种子，保证每次刷新网页样本不发生改变
    sampled_data = random.sample(
        test_samples, min(sample_size, len(test_samples)))
    return sampled_data, ref_by_score

# ================= 界面与交互模块 =================


def format_profile_ui(profile: dict):
    st.markdown(
        f"**性别**: {profile.get('gender', '未知')} | **年龄**: {profile.get('age', '未知')}")
    st.markdown(
        f"**职业**: {profile.get('occupation', '未知')} | **性格**: {', '.join(profile.get('personality', []))}")
    st.markdown(f"**背景**: {profile.get('background', '未知')}")
    st.markdown(f"**日常兴趣**: {', '.join(profile.get('daily_interests', []))}")
    st.markdown(f"**旅行习惯**: {', '.join(profile.get('travel_habits', []))}")
    st.markdown(
        f"**饮食偏好**: {', '.join(profile.get('dining_preferences', []))}")
    st.markdown(f"**消费习惯**: {', '.join(profile.get('spending_habits', []))}")
    st.markdown(f"**其他**: {', '.join(profile.get('other_aspects', []))}")


def compute_running_metrics(samples: list[dict], annotations: list[dict]) -> dict | None:
    if not annotations:
        return None

    sample_indices: list[int] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    for ann in annotations:
        idx = ann.get("sample_idx")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if idx_int < 0 or idx_int >= len(samples):
            continue

        sample_indices.append(idx_int)
        y_true.append(float(samples[idx_int]["gt_score"]))
        y_pred.append(float(ann["annotated_score"]))

    if not y_true:
        return None

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    r2 = None
    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)

    pearson_corr, spearman_corr = None, None
    if len(y_true) >= 2 and np.std(y_true) != 0 and np.std(y_pred) != 0:
        pearson_corr = float(pearsonr(y_true, y_pred)[0])
        spearman_corr = float(spearmanr(y_true, y_pred)[0])

    return {
        "n": len(y_true),
        "sample_indices": sample_indices,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": None if r2 is None else float(r2),
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }


def compute_running_reason_metrics(samples: list[dict], annotations: list[dict]) -> dict | None:
    if not annotations:
        return None

    y_true_reason: list[str] = []
    y_pred_reason: list[str] = []
    y_true_bin: list[int] = []
    y_pred_bin: list[int] = []

    for ann in annotations:
        idx = ann.get("sample_idx")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if idx_int < 0 or idx_int >= len(samples):
            continue

        gt_reason = str(samples[idx_int].get("gt_reason", ""))
        pred_reason = str(ann.get("annotated_reason", ""))
        if not gt_reason or not pred_reason:
            continue

        y_true_reason.append(gt_reason)
        y_pred_reason.append(pred_reason)

        y_true_bin.append(0 if gt_reason == "满意" else 1)   # 1 表示不满意
        y_pred_bin.append(0 if pred_reason == "满意" else 1)

    if not y_true_reason:
        return None

    # 二分类：满意 vs 不满意（把“不满意”作为正类=1）
    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    # 多分类：原因文本多分类（含“满意”）
    multi_acc = accuracy_score(y_true_reason, y_pred_reason)
    uniq_true = len(set(y_true_reason))
    uniq_pred = len(set(y_pred_reason))
    multi_p = multi_r = multi_f1 = None
    if uniq_true >= 2 and uniq_pred >= 2:
        multi_p, multi_r, multi_f1, _ = precision_recall_fscore_support(
            y_true_reason,
            y_pred_reason,
            average="macro",
            zero_division=0,
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true_reason,
            y_pred_reason,
            average="weighted",
            zero_division=0,
        )

    return {
        "n": len(y_true_reason),
        "bin": {
            "acc": float(bin_acc),
            "p": float(bin_p),
            "r": float(bin_r),
            "f1": float(bin_f1),
        },
        "multi": {
            "acc": float(multi_acc),
            "macro_p": None if multi_p is None else float(multi_p),
            "macro_r": None if multi_r is None else float(multi_r),
            "macro_f1": None if multi_f1 is None else float(multi_f1),
            "weighted_p": None if multi_p is None else float(weighted_p),
            "weighted_r": None if multi_r is None else float(weighted_r),
            "weighted_f1": None if multi_f1 is None else float(weighted_f1),
        },
    }


def get_annotator_save_path(annotator_name: str) -> str:
    # 保持原有文件命名风格（仅做路径分隔符安全替换），避免破坏已有数据加载。
    safe_name = (annotator_name or "").strip()
    for sep in [os.sep, getattr(os.path, "altsep", None)]:
        if sep:
            safe_name = safe_name.replace(sep, "_")
    return f"human_annotations/{safe_name}.json"


def load_annotations_for_annotator(annotator_name: str, total_samples: int) -> tuple[list[dict], int]:
    if not annotator_name.strip():
        return [], 0

    save_path = get_annotator_save_path(annotator_name)
    if not os.path.exists(save_path):
        return [], 0

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], 0

    results = data.get("results", [])
    if not isinstance(results, list):
        return [], 0

    # 用 sample_idx 去重/覆盖，保证最终与样本索引一一对应
    by_idx: dict[int, dict] = {}
    for ann in results:
        if not isinstance(ann, dict):
            continue
        idx = ann.get("sample_idx")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx_int < total_samples:
            by_idx[idx_int] = ann

    annotated_indices = sorted(by_idx.keys())
    next_idx = 0
    annotated_set = set(by_idx.keys())
    for i in range(total_samples):
        if i not in annotated_set:
            next_idx = i
            break
    else:
        next_idx = total_samples

    annotations = [by_idx[i] for i in annotated_indices]
    return annotations, next_idx


def save_annotations_for_annotator(annotator_name: str, annotations: list[dict], metrics: dict | None = None) -> None:
    if not annotator_name.strip():
        return

    os.makedirs("human_annotations", exist_ok=True)
    save_path = get_annotator_save_path(annotator_name)
    payload = {"annotator": annotator_name, "results": annotations}
    if metrics is not None:
        payload["metrics"] = metrics

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    st.set_page_config(page_title="满意度人工标注系统", layout="wide")
    st.title("🗣️ 满意度人工标注系统")

    # 1. 初始化 Session State
    if 'samples' not in st.session_state:
        samples, ref_by_score = load_and_sample_test_data(sample_size=100, ref_per_score=3)
        st.session_state.samples = samples
        st.session_state.ref_by_score = ref_by_score
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "标注"

    # 2. 侧边栏：标注者信息与进度
    with st.sidebar:
        st.header("📋 标注设置")
        annotator_name = st.text_input("请输入标注者姓名：", value="Annotator_01")

        # 允许不同标注者在同一浏览器会话中切换名字时，自动加载/恢复各自的中间结果
        if "loaded_annotator" not in st.session_state:
            st.session_state.loaded_annotator = None
        if st.session_state.loaded_annotator != annotator_name:
            total_samples = len(st.session_state.samples)
            loaded_annotations, loaded_next_idx = load_annotations_for_annotator(
                annotator_name,
                total_samples=total_samples,
            )
            st.session_state.annotations = loaded_annotations
            st.session_state.current_idx = loaded_next_idx
            st.session_state.loaded_annotator = annotator_name

        total_samples = len(st.session_state.samples)
        current_idx = st.session_state.current_idx

        st.progress(current_idx / total_samples if total_samples > 0 else 0)
        st.write(f"当前进度: **{current_idx} / {total_samples}**")

        running = compute_running_metrics(
            st.session_state.samples,
            st.session_state.get("annotations", []),
        )
        if running is not None:
            st.markdown("---")
            st.subheader("📈 截至目前指标")
            st.caption(f"已标注样本数: {running['n']}")

            c1, c2 = st.columns(2)
            c1.metric("MAE", f"{running['mae']:.4f}")
            c2.metric("RMSE", f"{running['rmse']:.4f}")

            c3, c4 = st.columns(2)
            c3.metric("R2", "N/A" if running["r2"] is None else f"{running['r2']:.4f}")
            c4.metric("Pearson", "N/A" if running["pearson"] is None else f"{running['pearson']:.4f}")

            c5, _ = st.columns(2)
            c5.metric("Spearman", "N/A" if running["spearman"] is None else f"{running['spearman']:.4f}")

        running_reason = compute_running_reason_metrics(
            st.session_state.samples,
            st.session_state.get("annotations", []),
        )
        if running_reason is not None:
            st.markdown("---")
            st.subheader("🏷️ 原因分类指标")
            st.caption(f"已计入原因的样本数: {running_reason['n']}")

            st.markdown("**满意/不满意（二分类，不满意为正类）**")
            b1, b2 = st.columns(2)
            b1.metric("Acc", f"{running_reason['bin']['acc']:.4f}")
            b2.metric("F1", f"{running_reason['bin']['f1']:.4f}")
            b3, b4 = st.columns(2)
            b3.metric("Precision", f"{running_reason['bin']['p']:.4f}")
            b4.metric("Recall", f"{running_reason['bin']['r']:.4f}")

            st.markdown("**原因多分类（含满意）**")
            m1, m2 = st.columns(2)
            m1.metric("Acc", f"{running_reason['multi']['acc']:.4f}")
            m2.metric(
                "Macro-F1",
                "N/A" if running_reason["multi"]["macro_f1"] is None else f"{running_reason['multi']['macro_f1']:.4f}",
            )
            m3, m4 = st.columns(2)
            m3.metric(
                "Macro-Precision",
                "N/A" if running_reason["multi"]["macro_p"] is None else f"{running_reason['multi']['macro_p']:.4f}",
            )
            m4.metric(
                "Macro-Recall",
                "N/A" if running_reason["multi"]["macro_r"] is None else f"{running_reason['multi']['macro_r']:.4f}",
            )
            m5, m6 = st.columns(2)
            m5.metric(
                "Weighted-F1",
                "N/A" if running_reason["multi"]["weighted_f1"] is None else f"{running_reason['multi']['weighted_f1']:.4f}",
            )
            m6.metric(
                "Weighted-Precision",
                "N/A" if running_reason["multi"]["weighted_p"] is None else f"{running_reason['multi']['weighted_p']:.4f}",
            )
            m7, m8 = st.columns(2)
            m7.metric(
                "Weighted-Recall",
                "N/A" if running_reason["multi"]["weighted_r"] is None else f"{running_reason['multi']['weighted_r']:.4f}",
            )

        st.markdown("---")
        st.session_state.active_page = st.radio(
            "页面",
            options=["标注", "参考"],
            index=0 if st.session_state.active_page == "标注" else 1,
        )

        if st.button("重置标注进度", type="secondary"):
            st.session_state.current_idx = 0
            st.session_state.annotations = []
            if annotator_name.strip():
                save_path = get_annotator_save_path(annotator_name)
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except Exception:
                        # 忽略删除失败：仍保留内存中的重置效果
                        pass
            st.session_state.loaded_annotator = None
            st.rerun()

    # 参考页：展示来自训练集的分数分层样本（不影响标注进度）
    if st.session_state.active_page == "参考":
        st.subheader("📚 参考样本（来自训练集，按分数分层抽取）")
        ref_by_score = st.session_state.get("ref_by_score", {i: [] for i in range(1, 6)})
        score_choice = st.selectbox("选择要查看的参考分数", options=[1, 2, 3, 4, 5], index=4)
        refs = ref_by_score.get(int(score_choice), [])
        if not refs:
            st.info("该分数在训练集中没有可用参考样本。")
            return

        for j, sample in enumerate(refs):
            with st.expander(f"参考样本 {j+1}（GT 分数：{sample['gt_score']}，任务：{sample.get('task','')}）", expanded=(j == 0)):
                st.caption(f"模型: {sample.get('chat_model', '')}")
                col_left, col_right = st.columns([1.5, 1])

                with col_left:
                    st.markdown(f"**🎯 任务背景**:\n\n{sample.get('task_context','')}")
                    st.markdown("---")
                    chat_container = st.container(height=420)
                    with chat_container:
                        for utt in sample.get('history', []):
                            role = "user" if utt.get('role') == "user" else "assistant"
                            with st.chat_message(role):
                                st.write(utt.get('content', ''))

                with col_right:
                    st.subheader("👤 用户画像与背景")
                    show_profile = st.checkbox(
                        "显示详细用户画像",
                        value=True,
                        key=f"ref_show_profile_{int(score_choice)}_{j}",
                    )
                    if show_profile:
                        format_profile_ui(sample.get('profile', {}))
        return

    # 3. 如果所有样本都标注完成了，展示评估结果
    if current_idx >= total_samples:
        st.success("🎉 所有标注任务已完成！感谢你的辛苦工作。")

        annotated_by_idx: dict[int, dict] = {
            int(ann.get("sample_idx")): ann
            for ann in st.session_state.annotations
            if isinstance(ann, dict) and ann.get("sample_idx") is not None
        }

        # 计算指标（按 sample_idx 对齐，避免annotations顺序与样本顺序不一致）
        y_true: list[float] = []
        y_pred: list[float] = []
        for i in range(total_samples):
            ann = annotated_by_idx.get(i)
            if not ann:
                continue
            y_true.append(float(st.session_state.samples[i]["gt_score"]))
            y_pred.append(float(ann["annotated_score"]))

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        score_acc = accuracy_score(y_true, y_pred)
        kappa_score = cohen_kappa_score(y_true, y_pred)
        binary_acc = accuracy_score([0 if s <= 3 else 1 for s in y_true], [0 if s <= 3 else 1 for s in y_pred])
        # 防止方差为0导致的皮尔逊计算警告
        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            pearson_corr, spearman_corr = 0.0, 0.0
        else:
            pearson_corr = pearsonr(y_true, y_pred)[0]
            spearman_corr = spearmanr(y_true, y_pred)[0]

        st.subheader("📊 人工标注与 Ground Truth 相关性评估")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("R2 Score", f"{r2:.4f}")
        col4.metric("Pearson", f"{pearson_corr:.4f}")
        col5.metric("Spearman", f"{spearman_corr:.4f}")
        col6, col7, col8 = st.columns(3)
        col6.metric("分数准确率", f"{score_acc:.4f}")
        col7.metric("Cohen's Kappa", f"{kappa_score:.4f}")
        col8.metric("满意/不满意二分类准确率", f"{binary_acc:.4f}")

        # 保存结果
        metrics = {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson_corr, "spearman": spearman_corr}
        save_path = get_annotator_save_path(annotator_name)
        save_annotations_for_annotator(annotator_name, st.session_state.annotations, metrics=metrics)
        st.info(f"💾 标注结果已保存至当前目录下的 `{save_path}`")
        return

    # 4. 展示当前标注样本
    sample = st.session_state.samples[current_idx]

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader(f"对话历史 (任务: {sample['task']})")
        st.caption(f"当前使用的模型: {sample['chat_model']}")

        # 使用 Streamlit 原生 Chat 气泡展示对话
        chat_container = st.container(height=580)
        with chat_container:
            for utt in sample['history']:
                role = "user" if utt['role'] == "user" else "assistant"
                with st.chat_message(role):
                    st.write(utt['content'])

    with col_right:
        st.subheader("👤 用户画像与背景")
        with st.expander("查看详细用户画像", expanded=True):
            format_profile_ui(sample['profile'])

        st.markdown("---")
        st.markdown(f"**🎯 任务背景**:\n\n{sample['task_context']}")

        st.markdown("---")
        st.subheader("✍️ 请进行标注")

        # 标注表单
        with st.form(key=f"annotation_form_{current_idx}"):
            score = st.slider("1. 满意度得分 (1-5)", min_value=1,
                              max_value=5, value=5, step=1)

            # 由于 Streamlit 表单内的组件不能动态交互，我们把原因输入直接放出来
            reason_options = ['满意', '不够多样', '不可用', '不够细致', '不满足需求', '其它']
            reason = st.selectbox(
                "2. 不满意原因 (如果得分 <= 3 才生效)",
                options=reason_options,
                index=0
            )

            submit_button = st.form_submit_button(label="提交并进入下一条 ➡️")

            if submit_button:
                if not annotator_name.strip():
                    st.error("请在左侧侧边栏输入标注者姓名！")
                else:
                    # 整理标注结果
                    final_reason = reason if score <= 3 else "满意"

                    annotation_record = {
                        "sample_idx": current_idx,
                        "task": sample['task'],
                        "annotated_score": score,
                        "annotated_reason": final_reason,
                        "gt_score": sample['gt_score'],
                        "gt_reason": sample['gt_reason'],
                        # 你也可以把原始文本存下来，方便后续追溯
                        "history_length": len(sample['history'])
                    }

                    st.session_state.annotations.append(annotation_record)
                    st.session_state.current_idx += 1
                    # 每次提交都立刻落盘，保证中间结果可恢复
                    save_annotations_for_annotator(annotator_name, st.session_state.annotations)
                    st.rerun()


if __name__ == "__main__":
    main()
