import os
import json

from basic_info import SIM_DIR, OUTPUT_DIR

dir_name = SIM_DIR
file_dir = OUTPUT_DIR
strategy_field = 'strategy'
rating_fields = ['Detail Level', 'Practical Usefulness', 'Diversity']
for task_list in [['new travel planning', 'preparing gifts', 'travel planning', 'recipe planning', 'skills learning planning']]:
    scores = {}
    for task in task_list:
        files = os.listdir(os.path.join(dir_name, task))
        files = [file for file in files if file.endswith('.json')]
        files.sort(key=lambda x: x.split('.')[0])
        for file in files:
            with open(os.path.join(dir_name, task, file), 'r') as f:
                data = json.load(f)
            strategy = data[strategy_field]['final']
            rating = data['rating']
            for key in strategy:
                if key not in scores:
                    scores[key] = {}
                if strategy[key] not in scores[key]:
                    scores[key][strategy[key]] = {}
                for field in rating_fields:
                    if field not in scores[key][strategy[key]]:
                        scores[key][strategy[key]][field] = []
                    scores[key][strategy[key]][field].append(rating[field])
    print(task_list)
    for key in scores:
        print(key)
        for field in rating_fields:
            print(field)
            print(*list(scores[key].keys()), sep='\t')
            for strategy in scores[key]:
                print(f'{sum(scores[key][strategy][field]) / len(scores[key][strategy][field]):.4f}', end='\t')
            print()
    print()
