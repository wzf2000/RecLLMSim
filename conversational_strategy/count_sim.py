import os
import json

from basic_info import SIM_DIR, OUTPUT_DIR

dir_name = SIM_DIR
file_dir = OUTPUT_DIR
strategy_field = 'strategy_V2'
heads = ['order_Depth', 'order_Breadth', 'order_DepthBreadth', 'order_BreadthDepth', 'feedback_NoFeedback', 'feedback_Positive', 'feedback_Negative', 'feedback_Both']
for task_list in [['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning'], ['new travel planning', 'travel planning'], ['recipe planning'], ['preparing gifts'], ['skills learning planning']]:
    counts = {}
    for task in task_list:
        files = os.listdir(os.path.join(dir_name, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(dir_name, task, file), 'r') as f:
                data = json.load(f)
            strategy = data[strategy_field]['final']
            for key in strategy:
                if key not in counts:
                    counts[key] = {}
                if strategy[key] not in counts[key]:
                    counts[key][strategy[key]] = 0
                counts[key][strategy[key]] += 1

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(os.path.join(file_dir, 'count_sim.txt')):
        with open(os.path.join(file_dir, 'count_sim.txt'), 'w') as f:
            f.write('\t'.join(heads) + '\n')
    with open(os.path.join(file_dir, 'count_sim.txt'), 'a') as f:
        column = []
        for col in heads:
            key, value = col.split('_')
            column.append(str(counts[key].get(value, 0)))
        f.write('\t'.join(column) + '\n')
