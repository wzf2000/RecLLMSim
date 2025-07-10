import os
import json
import streamlit as st

from output_format import (
    InformationRequest, Context, Question, OrderV2, Feedback, Rating,
    Explanation, Promise, Politeness, Formality
)
from basic_info import SIM_DIR, HUMAN_DIR

def label_single(history: list[dict[str, str]], version: int, human: bool = False, ground_truth: dict[str, dict] = {}) -> dict[str, str]:
    user_only = st.sidebar.checkbox('User Only')
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.container(height=500):
            st.markdown('### History')
            for utt in history:
                if user_only and utt['role'] == 'assistant':
                    continue
                st.chat_message(utt['role']).markdown(utt['content'] if human else utt['content_zh'])
                if not human:
                    st.chat_message(utt['role']).markdown(utt['content'])
    with col2:
        st.markdown('### Strategy')
        strategy = {}
        strategy['final'] = {}
        if version == 1:
            strategy['final']['information_request'] = st.selectbox('Planning', [InformationRequest.Planning.value, InformationRequest.Sequential.value])
            strategy['final']['context'] = st.selectbox('Context', [Context.High.value, Context.Low.value])
            strategy['final']['question'] = st.selectbox('Specificity', [Question.Broad.value, Question.Specific.value])
        elif version == 2:
            strategy['final']['order'] = st.selectbox('Order', [OrderV2.Depth.value, OrderV2.Breadth.value, OrderV2.DepthBreadth.value, OrderV2.BreadthDepth.value])
            strategy['final']['feedback'] = st.selectbox('Feedback', [Feedback.NoFeedback.value, Feedback.Positive.value, Feedback.Negative.value, Feedback.Both.value])
        elif version == 3:
            strategy['final']['question_broadness'] = st.select_slider('Question Broadness', options=[Rating.One.value, Rating.Two.value, Rating.Three.value, Rating.Four.value, Rating.Five.value])
            strategy['final']['context_dependency'] = st.select_slider('Context Dependency', options=[Rating.One.value, Rating.Two.value, Rating.Three.value, Rating.Four.value, Rating.Five.value])
            strategy['final']['feedback'] = st.selectbox('Feedback', [Feedback.NoFeedback.value, Feedback.Positive.value, Feedback.Negative.value, Feedback.Both.value])
        elif version == 4:
            strategy['final']['explanation'] = st.selectbox('Explanation', [Explanation.Frequent.value, Explanation.Rare.value, Explanation.NoExplanation.value])
            strategy['final']['promise'] = st.selectbox('Promise', [Promise.HavePromise.value, Promise.NoPromise.value])
            strategy['final']['politeness'] = st.selectbox('Politeness', [Politeness.Polite.value, Politeness.Neutral.value, Politeness.Impolite.value])
            strategy['final']['formality'] = st.selectbox('Formality', [Formality.Oral.value, Formality.Formal.value])
        if st.button('Show GT'):
            st.markdown('### Ground Truth')
            for key, value in ground_truth['final'].items():
                st.markdown(f'{key}: {value}')
        if st.button('Submit'):
            return strategy
        else:
            return {}

def label_sim(version: int, strategy_name: str, compare_name: str, sample: int):
    dir_name = SIM_DIR
    task_list = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']

    task = st.selectbox('Task', task_list)
    assert task in task_list, 'Task not found'

    files = os.listdir(os.path.join(dir_name, task))
    files = [file for file in files if file.endswith('.json')]
    files.sort(key=lambda x: x.split('.')[0])
    files = files[:sample] if sample > 0 else files

    right = {}

    for i, file in enumerate(files):
        with open(os.path.join(dir_name, task, file), 'r') as f:
            data = json.load(f)
        if strategy_name in data:
            for key in data[strategy_name]['final']:
                if key not in right:
                    right[key] = []
                right[key].append(data[strategy_name]['final'][key] == data[compare_name]['final'][key])
            continue
        st.progress(i / len(files), f'{i} / {len(files)}')
        data[strategy_name] = label_single(data['history'], version, False, data[compare_name])
        if data[strategy_name] != {}:
            with open(os.path.join(dir_name, task, file), 'w') as fw:
                json.dump(data, fw, ensure_ascii=False, indent=4)
            st.rerun()
        return

    for key in right:
        st.markdown(f'{key}: {sum(right[key]) / len(right[key])} ({sum(right[key])} / {len(right[key])})')
        # output wrong files
        wrong_files = [files[i] for i in range(len(files)) if not right[key][i]]
        if wrong_files:
            st.markdown(f'Wrong files for {key}:')
            for wrong_file in wrong_files:
                st.markdown(f'- {wrong_file}')

def label_human(version: int, strategy_name: str, compare_name: str, sample: int):
    dir_name1 = HUMAN_DIR
    dir_name2 = os.path.join(HUMAN_DIR, '..', 'human_exp_V2')
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    users1 = os.listdir(dir_name1)
    users1 = [user for user in users1 if os.path.isdir(os.path.join(dir_name1, user))]
    users1.sort()
    users2 = os.listdir(dir_name2)
    users2 = [user for user in users2 if os.path.isdir(os.path.join(dir_name2, user))]
    users2.sort()

    part1_len = len(users1)
    part2_len = len(users2)
    belongs = [1] * part1_len + [2] * part2_len
    users = users1 + users2
    # shuffule users & belongs
    import random
    random.seed(42)  # For reproducibility
    combined = list(zip(users, belongs))
    random.shuffle(combined)
    users, belongs = zip(*combined)

    task = st.selectbox('Task', task_list)
    assert task in task_list, 'Task not found'

    users = users[:sample] if sample > 0 else users
    belongs = belongs[:sample] if sample > 0 else belongs

    right = {}
    errors = {}

    for i, (user, belong) in enumerate(zip(users, belongs)):
        if belong == 1:
            dir_name = dir_name1
        else:
            dir_name = dir_name2
        if not os.path.exists(os.path.join(dir_name, user, task)):
            continue
        files = os.listdir(os.path.join(dir_name, user, task))
        if sample:
            files = files[:1]
        if len(files) == 0:
            continue
        for file in files:
            with open(os.path.join(dir_name, user, task, file), 'r') as f:
                data = json.load(f)
            if strategy_name in data:
                for key in data[strategy_name]['final']:
                    if key not in right:
                        right[key] = []
                    if data[strategy_name]['final'][key] != data[compare_name]['final'][key]:
                        st.markdown(f'{user} {task} {file} {key} {data[strategy_name]["final"][key]} {data[compare_name]["final"][key]}')
                    right[key].append(data[strategy_name]['final'][key] == data[compare_name]['final'][key])
                    if isinstance(data[strategy_name]['final'][key], int):
                        if key not in errors:
                            errors[key] = []
                        errors[key].append(abs(data[strategy_name]['final'][key] - data[compare_name]['final'][key]))
                continue
            st.progress(i / len(users), f'{i} / {len(users)}')
            data[strategy_name] = label_single(data['history'], version, True, data[compare_name])
            if data[strategy_name] != {}:
                with open(os.path.join(dir_name, user, task, file), 'w') as fw:
                    json.dump(data, fw, ensure_ascii=False, indent=4)
                st.rerun()
            return

    for key in right:
        st.markdown(f'{key}: {sum(right[key]) / len(right[key])} ({sum(right[key])} / {len(right[key])})')
        # output wrong files
        wrong_files = [users[i] + ' ' + task + ' ' + files[i] for i in range(len(users)) if not right[key][i]]
        if wrong_files:
            st.markdown(f'Wrong files for {key}:')
            for wrong_file in wrong_files:
                st.markdown(f'- {wrong_file}')
    for key in errors:
        st.markdown(f'{key} MAE: {sum(errors[key]) / len(errors[key])} ({sum(errors[key])} / {len(errors[key])})')
        st.markdown(f'{key} MSE: {sum([error ** 2 for error in errors[key]]) / len(errors[key])} ({sum([error ** 2 for error in errors[key]])} / {len(errors[key])})')

def main():
    st.set_page_config(page_title='Human Label', layout='wide')
    st.sidebar.markdown('### Version')
    version = st.sidebar.selectbox('Version', [1, 2, 3, 4])
    assert version in [1, 2, 3, 4], 'Version not supported'
    sample = st.sidebar.number_input('Sample Size', min_value=0, max_value=2500, value=10, step=1)
    assert sample >= 0, 'Sample size must be non-negative'

    st.sidebar.markdown('### Mode')
    mode = st.sidebar.selectbox('Mode', ['Simulation', 'Human'])
    strategy_name = 'strategy_human' if version == 1 else 'strategy_human_V2' if version == 2 else 'strategy_human_V3' if version == 3 else 'strategy_human_V4'
    compare_name = 'strategy' if version == 1 else 'strategy_V2' if version == 2 else 'strategy_V3' if version == 3 else 'strategy_V4'

    if mode == 'Simulation':
        label_sim(version, strategy_name, compare_name, int(sample))
    elif mode == 'Human':
        label_human(version, strategy_name, compare_name, int(sample))

if __name__ == '__main__':
    main()
