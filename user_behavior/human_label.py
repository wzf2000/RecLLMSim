import os
import json
import streamlit as st

from output_format import (
    ProblemSolving, Order, Feedback, Rating,
    Explanation, Promise, Politeness, Formality, Usefulness
)
from basic_info import SIM_DIR, HUMAN_DIR

def get_selectbox(label: str, original_option: str | None, options: list[str]) -> str | None:
    if original_option is None:
        return st.selectbox(label, options, index=0)
    else:
        if original_option not in options:
            return st.selectbox(label, options, index=0)
        else:
            return st.selectbox(label, options, index=options.index(original_option))

def get_select_slider(label: str, original_option: int | None, options: list[int]) -> int | tuple[int, int]:
    if original_option is None:
        return st.select_slider(label, options=options, value=options[0])
    else:
        if original_option not in options:
            return st.select_slider(label, options=options, value=options[0])
        else:
            return st.select_slider(label, options=options, value=original_option)

def label_with_version(version: int, original_labels: dict | None = None) -> dict:
    # original_labels looks like:
    # original_labels = {"XXX": "YYY", ...}, XXX is the label name, YYY is the label value for now
    # set the default values for each version
    if original_labels is None:
        original_labels = {}
    else:
        with st.expander('Original Labels', expanded=True):
            st.json(original_labels)
    strategy = {}
    strategy['final'] = {}
    if version == 1:
        strategy['final']['problem_solving'] = get_selectbox('Problem Solving', original_labels.get('problem_solving'), [ProblemSolving.AllInOne.value, ProblemSolving.StepByStep.value])
        strategy['final']['order'] = get_selectbox('Order', original_labels.get('order'), [Order.Depth.value, Order.Breadth.value, Order.DepthBreadth.value, Order.BreadthDepth.value])
    elif version == 2:
        strategy['final']['question_broadness'] = get_select_slider('Question Broadness', original_labels.get('question_broadness'), [Rating.One.value, Rating.Two.value, Rating.Three.value, Rating.Four.value, Rating.Five.value])
        strategy['final']['context_dependency'] = get_select_slider('Context Dependency', original_labels.get('context_dependency'), [Rating.One.value, Rating.Two.value, Rating.Three.value, Rating.Four.value, Rating.Five.value])
    elif version == 3:
        strategy['final']['feedback'] = get_selectbox('Feedback', original_labels.get('feedback'), [Feedback.NoFeedback.value, Feedback.Positive.value, Feedback.Negative.value, Feedback.Both.value])
        strategy['final']['explanation'] = get_selectbox('Explanation', original_labels.get('explanation'), [Explanation.Frequent.value, Explanation.Rare.value, Explanation.NoExplanation.value])
        strategy['final']['promise'] = get_selectbox('Promise', original_labels.get('promise'), [Promise.HavePromise.value, Promise.NoPromise.value])
        strategy['final']['politeness'] = get_selectbox('Politeness', original_labels.get('politeness'), [Politeness.Polite.value, Politeness.Neutral.value, Politeness.Impolite.value])
        strategy['final']['formality'] = get_selectbox('Formality', original_labels.get('formality'), [Formality.Oral.value, Formality.Formal.value, Formality.Mixed.value])
    elif version == 4:
        strategy['final']['utility'] = get_selectbox('Utility', original_labels.get('utility'), [Usefulness.Low.value, Usefulness.Moderate.value, Usefulness.High.value])
        strategy['final']['operability'] = get_selectbox('Operability', original_labels.get('operability'), [Usefulness.Low.value, Usefulness.Moderate.value, Usefulness.High.value])
    return strategy

def label_single(history: list[dict[str, str]], version: int, human: bool = False, ground_truth: dict[str, dict] = {}, original_labels: dict | None = None) -> dict[str, str]:
    user_only = st.sidebar.checkbox('User Only', value=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.container(height=600 if 'height' not in st.session_state else st.session_state['height']):
            st.markdown('### History')
            for utt in history:
                if user_only and utt['role'] == 'assistant':
                    continue
                st.chat_message(utt['role']).markdown(utt['content'] if human else utt['content_zh'])
                if not human:
                    st.chat_message(utt['role']).markdown(utt['content'])
    with col2:
        strategy = label_with_version(version, original_labels)
        if st.button('Submit'):
            return strategy
        else:
            return {}

def get_sim_files(sample: int, all: bool = False) -> list[str]:
    if all:
        tasks = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    else:
        task_list = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
        task = st.selectbox('Task', task_list)
        assert task in task_list, 'Task not found'
        tasks = [task]
    ret = []

    def get_single_task_files(task: str) -> list[str]:
        files = os.listdir(os.path.join(SIM_DIR, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        files = files[:sample] if sample > 0 else files
        files = [os.path.join(SIM_DIR, task, file) for file in files]
        return files

    for task in tasks:
        files = get_single_task_files(task)
        ret.extend(files)
    return ret

def get_human_files(sample: int, all: bool = False) -> list[str]:
    dir_name1 = HUMAN_DIR
    dir_name2 = os.path.join(HUMAN_DIR, '..', 'human_exp_V2')
    if all:
        tasks = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    else:
        task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
        task = st.selectbox('Task', task_list)
        assert task in task_list, 'Task not found'
        tasks = [task]
    ret = []

    def get_single_task_files(task: str) -> list[str]:
        users1 = os.listdir(dir_name1)
        users1 = [user for user in users1 if os.path.isdir(os.path.join(dir_name1, user)) and os.path.exists(os.path.join(dir_name1, user, task))]
        users1.sort()
        users2 = os.listdir(dir_name2)
        users2 = [user for user in users2 if os.path.isdir(os.path.join(dir_name2, user)) and os.path.exists(os.path.join(dir_name2, user, task))]
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

        files = [os.listdir(os.path.join(dir_name1 if belong == 1 else dir_name2, user, task)) for user, belong in zip(users, belongs)]
        files = [os.path.join(dir_name1 if belong == 1 else dir_name2, user, task, file[0]) for user, belong, file in zip(users, belongs, files) if len(file) > 0]
        files = files[:sample] if sample > 0 else files
        return files

    for task in tasks:
        files = get_single_task_files(task)
        ret.extend(files)
    return ret

def get_label_keys(version: int) -> list[str]:
    if version == 1:
        return ['problem_solving', 'order']
    elif version == 2:
        return ['question_broadness', 'context_dependency']
    elif version == 3:
        return ['feedback', 'explanation', 'promise', 'politeness', 'formality']
    elif version == 4:
        return ['utility', 'operability']
    else:
        raise ValueError(f'Unsupported version: {version}')

def check_fully_labeled(data: dict, strategy_name: str, version: int):
    if strategy_name not in data:
        return False
    if 'final' not in data[strategy_name]:
        return False
    required_keys = get_label_keys(version)
    for key in required_keys:
        if key not in data[strategy_name]['final'] or data[strategy_name]['final'][key] is None:
            return False
    return True

def label_sim(version: int, strategy_name: str, compare_name: str, sample: int):
    files = get_sim_files(sample)
    right = {}

    for i, file in enumerate(files):
        with open(file, 'r') as f:
            data = json.load(f)
        if check_fully_labeled(data, strategy_name, version):
            for key in data[strategy_name]['final']:
                if key not in right:
                    right[key] = []
                right[key].append(data[strategy_name]['final'][key] == data[compare_name]['final'][key])
            continue
        st.progress(i / len(files), f'{i} / {len(files)}')
        data[strategy_name] = label_single(data['history'], version, False, data[compare_name], data.get(strategy_name, {}).get('final', {}))
        if data[strategy_name] != {}:
            with open(file, 'w') as fw:
                json.dump(data, fw, ensure_ascii=False, indent=4)
            st.rerun()
        return

    for key in get_label_keys(version):
        st.markdown(f'{key}: {sum(right[key]) / len(right[key])} ({sum(right[key])} / {len(right[key])})')
        # output wrong files
        wrong_files = [files[i] for i in range(len(files)) if not right[key][i]]
        if wrong_files:
            st.markdown(f'Wrong files for {key}:')
            for wrong_file in wrong_files:
                st.markdown(f'- {os.path.relpath(wrong_file)}')

def label_human(version: int, strategy_name: str, compare_name: str, sample: int):
    files = get_human_files(sample)
    right = {}
    errors = {}
    labeled_files = []

    for i, file in enumerate(files):
        with open(file, 'r') as f:
            data = json.load(f)

        if check_fully_labeled(data, strategy_name, version):
            labeled_files.append(os.path.relpath(file))
            for key in data[strategy_name]['final']:
                if key not in right:
                    right[key] = []
                right[key].append(data[strategy_name]['final'][key] == data[compare_name]['final'][key])
                if isinstance(data[strategy_name]['final'][key], int):
                    if key not in errors:
                        errors[key] = []
                    errors[key].append(abs(data[strategy_name]['final'][key] - data[compare_name]['final'][key]))
            continue
        st.progress(i / len(files), f'{i} / {len(files)}')
        data[strategy_name] = label_single(data['history'], version, True, data[compare_name], data.get(strategy_name, {}).get('final', {}))
        if data[strategy_name] != {}:
            with open(file, 'w') as fw:
                json.dump(data, fw, ensure_ascii=False, indent=4)
            st.rerun()
        return

    for key in get_label_keys(version):
        st.markdown(f'{key}: {sum(right[key]) / len(right[key])} ({sum(right[key])} / {len(right[key])})')
        # output wrong files
        wrong_files = [labeled_files[i] for i in range(len(files)) if not right[key][i]]
        if wrong_files:
            st.markdown(f'Wrong files for {key}:')
            for wrong_file in wrong_files:
                st.markdown(f'- {wrong_file}')
    for key in errors:
        st.markdown(f'{key} MAE: {sum(errors[key]) / len(errors[key])} ({sum(errors[key])} / {len(errors[key])})')
        st.markdown(f'{key} MSE: {sum([error ** 2 for error in errors[key]]) / len(errors[key])} ({sum([error ** 2 for error in errors[key]])} / {len(errors[key])})')

def label_in_order(version: int, mode: str):
    sample = st.sidebar.number_input('Sample Size', min_value=0, max_value=2500, value=10, step=1)
    strategy_name = 'strategy_human' if version == 1 else f'strategy_human_V{version}'
    compare_name = 'strategy_V1' if version == 1 else f'strategy_V{version}'
    assert sample >= 0, 'Sample size must be non-negative'
    if mode == 'Simulation':
        label_sim(version, strategy_name, compare_name, int(sample))
    elif mode == 'Human':
        label_human(version, strategy_name, compare_name, int(sample))

def get_labeled_files(files: list[str], strategy_name: str, version: int) -> list[str]:
    labeled_files = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        if check_fully_labeled(data, strategy_name, version):
            labeled_files.append(os.path.relpath(file))
    return labeled_files

def show_all_accuracy(files: list[str], strategy_name: str, compare_name: str):
    right = {}
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        if strategy_name not in data or compare_name not in data:
            st.error(f'Strategy {strategy_name} or {compare_name} not found in {file}')
            continue
        for key in data[strategy_name]['final']:
            if key == 'feedback':
                print(strategy_name, file)
            if key not in right:
                right[key] = []
            right[key].append(data[strategy_name]['final'][key] == data[compare_name]['final'][key])
    st.sidebar.markdown('### Overall Accuracy')
    for key in right:
        st.sidebar.markdown(f'{key}: {sum(right[key]) / len(right[key])} ({sum(right[key])} / {len(right[key])})')

def check_labeled_files(version: int, mode: str):
    sample = st.sidebar.number_input('Sample Size', min_value=0, max_value=2500, value=10, step=1)
    sample = int(sample)
    strategy_name = f'strategy_human_V{version}'
    compare_name = f'strategy_V{version}'
    if mode == 'Simulation':
        files = get_sim_files(sample)
    elif mode == 'Human':
        files = get_human_files(sample)
    else:
        st.error('Mode not supported')
        return
    all_task_files = get_sim_files(sample, all=True) + get_human_files(sample, all=True)
    all_task_files = get_labeled_files(all_task_files, strategy_name, version)
    show_all_accuracy(all_task_files, strategy_name, compare_name)
    files = get_labeled_files(files, strategy_name, version)
    file = st.selectbox('Files', files, index=0)
    if file:
        user_only = st.sidebar.checkbox('User Only', value=True)
        col1, col2 = st.columns([3, 1])
        with open(file, 'r') as f:
            data = json.load(f)
        with col1:
            st.markdown('### History')
            for utt in data['history']:
                if user_only and utt['role'] == 'assistant':
                    continue
                st.chat_message(utt['role']).markdown(utt['content'])
                if mode == 'Simulation':
                    st.chat_message(utt['role']).markdown(utt['content_zh'])
        with col2:
            st.markdown('### Auto-labeled Strategy')
            if compare_name in data:
                st.json(data[compare_name]['final'])
            else:
                st.error(f'Strategy {compare_name} not found in {file}')
            st.markdown('### Update Strategy')
            strategy = label_with_version(version, data.get(strategy_name, {}).get('final', {}))
            if st.button('Update'):
                data[strategy_name] = strategy
                with open(file, 'w') as fw:
                    json.dump(data, fw, ensure_ascii=False, indent=4)
                st.success('Strategy updated successfully')
                st.rerun()

def main():
    st.set_page_config(page_title='Human Label', layout='wide')
    st.sidebar.markdown('### Pipeline')
    pipe = st.sidebar.selectbox('Pipeline', ['Label in order', 'Check labeled files'])
    st.session_state['height'] = st.sidebar.number_input('Height', min_value=600, max_value=2000, value=600, step=50)
    st.sidebar.markdown('### Version')
    version = st.sidebar.selectbox('Version', [1, 2, 3, 4])
    assert version in [1, 2, 3, 4], 'Version not supported'

    st.sidebar.markdown('### Mode')
    mode = st.sidebar.selectbox('Mode', ['Simulation', 'Human'])
    assert mode in ['Simulation', 'Human'], 'Mode not supported'
    if pipe == 'Label in order':
        label_in_order(version, mode)
    elif pipe == 'Check labeled files':
        check_labeled_files(version, mode)
    else:
        st.error('Mode not supported')

if __name__ == '__main__':
    main()
