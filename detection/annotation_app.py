import os
import json
import random
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

# 假设你的获取数据函数保存在 metric_statistics.py 中
from metric_statistics import get_satisfaction_data
from data_split import split_by_user_group_shuffle_split

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

        total_samples = len(st.session_state.samples)
        current_idx = st.session_state.current_idx

        st.progress(current_idx / total_samples if total_samples > 0 else 0)
        st.write(f"当前进度: **{current_idx} / {total_samples}**")

        st.markdown("---")
        st.session_state.active_page = st.radio(
            "页面",
            options=["标注", "参考"],
            index=0 if st.session_state.active_page == "标注" else 1,
        )

        if st.button("重置标注进度", type="secondary"):
            st.session_state.current_idx = 0
            st.session_state.annotations = []
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
                st.markdown(f"**🎯 任务背景**:\n\n{sample.get('task_context','')}")
                st.markdown("---")
                chat_container = st.container(height=420)
                with chat_container:
                    for utt in sample.get('history', []):
                        role = "user" if utt.get('role') == "user" else "assistant"
                        with st.chat_message(role):
                            st.write(utt.get('content', ''))
        return

    # 3. 如果所有样本都标注完成了，展示评估结果
    if current_idx >= total_samples:
        st.success("🎉 所有标注任务已完成！感谢你的辛苦工作。")

        # 计算指标
        y_true = [s['gt_score'] for s in st.session_state.samples]
        y_pred = [ann['annotated_score'] for ann in st.session_state.annotations]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
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

        # 保存结果
        save_path = f"human_annotations/{annotator_name}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            output_data = {
                "annotator": annotator_name,
                "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson_corr, "spearman": spearman_corr},
                "results": st.session_state.annotations
            }
            json.dump(output_data, f, ensure_ascii=False, indent=2)

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
                    st.rerun()


if __name__ == "__main__":
    main()
