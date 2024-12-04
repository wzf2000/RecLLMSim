import os
import json

from basic_info import HUMAN_DIR, OUTPUT_DIR

dir_name = HUMAN_DIR
file_dir = OUTPUT_DIR
users = os.listdir(dir_name)
strategy_field = 'strategy_V2'
rating_fields = ['Detail Level', 'Practical Usefulness', 'Diversity']
for task_list in [['旅行规划', '礼物准备', '菜谱规划', '技能学习规划']]:
    scores = {}
    for user in users:
        for task in task_list:
            if not os.path.exists(os.path.join(dir_name, user, task)):
                continue
            files = os.listdir(os.path.join(dir_name, user, task))
            files = [file for file in files if file.endswith('.json')]
            files.sort(key=lambda x: x.split('.')[0])
            for file in files:
                with open(os.path.join(dir_name, user, task, file), 'r') as f:
                    data = json.load(f)
                strategy = data[strategy_field]['final']
                
                def convert_score(option: str) -> int:
                    if option == '详细':
                        return 2
                    elif option == '一般':
                        return 1
                    elif option == '简略':
                        return 0
                    elif option == '很有用':
                        return 2
                    elif option == '有一定参考':
                        return 1
                    elif option == '不可用':
                        return 0
                    elif option == '结果多样':
                        return 2
                    elif option == '结果单一':
                        return 0
                    else:
                        raise ValueError(f'Unknown option: {option}')
                
                questions = data['questionnaire']
                rating = {}
                for question in questions:
                    if question['summary'] == 'Detail':
                        rating_field = 'Detail Level'
                    elif question['summary'] == 'Availability':
                        rating_field = 'Practical Usefulness'
                    elif question['summary'] == 'Diversity':
                        rating_field = 'Diversity'
                    else:
                        continue
                    rating[rating_field] = convert_score(question['option'])
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
