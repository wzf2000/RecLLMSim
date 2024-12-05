import os
import json
import streamlit as st

from output_format import InformationRequest, Context, Question, OrderV2, Feedback
from basic_info import SIM_DIR, HUMAN_DIR

def label_single(history: list[dict[str, str]], version: int, human: bool = False) -> dict[str, str]:
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.container(height=500):
            st.markdown('### History')
            for utt in history:
                st.chat_message(utt['role']).markdown(utt['content'] if human else utt['content_zh'])
    with col2:
        st.markdown('### Strategy')
        strategy = {}
        strategy['final'] = {}
        if version == 1:
            strategy['final']['planning'] = st.selectbox('Planning', [InformationRequest.Planning.value, InformationRequest.Sequential.value])
            strategy['final']['context'] = st.selectbox('Context', [Context.High.value, Context.Low.value])
            strategy['final']['question'] = st.selectbox('Specificity', [Question.Broad.value, Question.Specific.value])
        elif version == 2:
            strategy['final']['order'] = st.selectbox('Order', [OrderV2.Depth.value, OrderV2.Breadth.value, OrderV2.DepthBreadth.value, OrderV2.BreadthDepth.value])
            strategy['final']['feedback'] = st.selectbox('Feedback', [Feedback.NoFeedback.value, Feedback.Positive.value, Feedback.Negative.value, Feedback.Both.value])
        if st.button('Submit'):
            return strategy
        else:
            return {}

def label_sim(version: int, strategy_name: str, compare_name: str, sample: bool = False):
    dir_name = SIM_DIR
    task_list = ['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']
    
    task = st.selectbox('Task', task_list)

    files = os.listdir(os.path.join(dir_name, task))
    files = [file for file in files if file.endswith('.json')]
    files.sort(key=lambda x: x.split('.')[0])
    if sample:
        files = files[:10]
    
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
        data[strategy_name]= label_single(data['history'], version, False)
        if data[strategy_name] != {}:
            with open(os.path.join(dir_name, task, file), 'w') as fw:
                json.dump(data, fw, ensure_ascii=False, indent=4)
            st.rerun()
        return

    for key in right:
        st.markdown(f'{key}: {sum(right[key]) / len(right[key])} ({sum(right[key])} / {len(right[key])})')

def label_human(version: int, strategy_name: str, compare_name: str, sample: bool = False):
    dir_name = HUMAN_DIR
    task_list = ['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']
    users = os.listdir(dir_name)
    users.sort()
    
    task = st.selectbox('Task', task_list)
    
    users = [user for user in users if os.path.exists(os.path.join(dir_name, user, task))]
    if sample:
        users = users[:10]
    
    right = {}
    
    for i, user in enumerate(users):
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
                continue
            st.progress(i / len(users), f'{i} / {len(users)}')
            data[strategy_name]= label_single(data['history'], version, True)
            if data[strategy_name] != {}:
                with open(os.path.join(dir_name, user, task, file), 'w') as fw:
                    json.dump(data, fw, ensure_ascii=False, indent=4)
                st.rerun()
            return

    for key in right:
        st.markdown(f'{key}: {sum(right[key]) / len(right[key])} ({sum(right[key])} / {len(right[key])})')

def main():
    st.set_page_config(page_title='Human Label', layout='wide')
    st.sidebar.markdown('### Version')
    version = st.sidebar.selectbox('Version', [1, 2])
    sample = st.sidebar.checkbox('Sample')
    
    st.sidebar.markdown('### Mode')
    mode = st.sidebar.selectbox('Mode', ['Simulation', 'Human'])
    strategy_name = 'strategy_human' if version == 1 else 'strategy_human_V2'
    compare_name = 'strategy' if version == 1 else 'strategy_V2'
    
    if mode == 'Simulation':
        label_sim(version, strategy_name, compare_name, sample)
    elif mode == 'Human':
        label_human(version, strategy_name, compare_name, sample)

if __name__ == '__main__':
    main()
