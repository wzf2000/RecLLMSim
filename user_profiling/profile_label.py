import os
import json
import numpy as np
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

from evaluate_util import compute_metrics
from pipe_util import split_train_test
from data_util import ModelType, get_human_data

output_dir = os.path.join(os.path.dirname(__file__), 'profile_labeled')

def get_task(attribute: str):
    if attribute == 'personality':
        return None
    elif attribute == 'daily_interests':
        return '技能学习规划'
    elif attribute == 'travel_habits':
        return '旅行规划'
    elif attribute == 'dining_preferences':
        return '菜谱规划'
    elif attribute == 'spending_habits':
        return '礼物准备'
    else:
        raise ValueError(f'Invalid attribute: {attribute}')

def get_all_data(attribute: str, label_id: str):
    X_human, y_human = get_human_data(attribute, get_task(attribute), model_type=ModelType.HUMAN)
    mlb = MultiLabelBinarizer()
    y_human = mlb.fit_transform(y_human)
    _, X_test, _, y_test = split_train_test(X_human, y_human)
    predictions = [[] for _ in range(len(X_test))]
    os.makedirs(os.path.join(output_dir, label_id, 'human_exp'), exist_ok=True)
    output_file = os.path.join(output_dir, label_id, 'human_exp', f'{attribute}.json')
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
    else:
        with open(output_file, 'r', encoding='utf8') as f:
            predictions = json.load(f)
    return X_test, y_test, mlb.classes_, predictions

def locate_unlabeled_profile(predictions: list[list[str]]):
    for i, prediction in enumerate(predictions):
        if len(prediction) < 3:
            return i
    return len(predictions)

def rank2prob(classes: np.ndarray, predictions: list[list[str]]) -> np.ndarray:
    probs = np.zeros((len(predictions), len(classes)))
    classes: list[str] = classes.tolist()
    for i, prediction in enumerate(predictions):
        for j, label in enumerate(prediction):
            probs[i][classes.index(label)] = 1.0 - j * 0.01
    return probs

def evaluate(y_test: np.ndarray, predictions: list[list[str]], classes: np.ndarray):
    probs = rank2prob(classes, predictions)
    return compute_metrics(y_test, probs, ranking_only=True)

def main():
    st.set_page_config(page_title='Profile Label', layout='wide')
    st.sidebar.markdown('### Profile Label')
    attribute = st.sidebar.selectbox('Attribute', ['personality', 'daily_interests', 'travel_habits', 'dining_preferences', 'spending_habits'])
    label_id = st.sidebar.selectbox('Label ID', ['A', 'B', 'C', 'D', 'E'])
    height = st.sidebar.slider('Height', 100, 1000, 600, step=25)
    show_assistant = st.sidebar.checkbox('Show Assistant', value=True)
    X_test, y_test, classes, predictions = get_all_data(attribute, label_id)
    location = locate_unlabeled_profile(predictions)
    st.progress(location / len(predictions), text=f'Labeled ratio: {location} / {len(predictions)} = {location / len(predictions):.2%}')
    col1, col2 = st.columns([4, 1])
    if location > 0:
        metrics = evaluate(y_test[:location], predictions[:location], classes)
        with col2:
            st.markdown('### Metrics')
            st.markdown(f'- **Hit Rate@1**: {metrics["hit_rate_1"]:.2%}')
            st.markdown(f'- **Hit Rate@3**: {metrics["hit_rate_3"]:.2%}')
            st.markdown(f'- **Recall@1**: {metrics["recall_1"]:.2%}')
            st.markdown(f'- **Recall@3**: {metrics["recall_3"]:.2%}')
    if location == len(predictions):
        st.success('All profiles have been labeled!')
        return
    conversation = X_test[location]
    with col1:
        with st.container(border=True, height=height):
            st.markdown("### Conversation")
            for utt in conversation:
                if not show_assistant and utt['role'] == 'assistant':
                    continue
                st.chat_message(utt['role']).markdown(utt['content'])
    attribute_name = ' '.join(list(map(lambda x: x.capitalize(), attribute.split('_'))))
    with col2:
        st.markdown('### Label')
        choices = st.multiselect(f'Choose from highest to lowest probability for **{attribute_name}**', classes, key=attribute, )
        if st.button('Submit'):
            if len(choices) < 3:
                st.error('Please choose at least three labels!')
                return
            predictions[location] = choices
            output_file = os.path.join(output_dir, label_id, 'human_exp', f'{attribute}.json')
            with open(output_file, 'w', encoding='utf8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=4)
            st.success('Submitted successfully!')
            st.rerun()

if __name__ == '__main__':
    main()
