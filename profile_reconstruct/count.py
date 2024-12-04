import os
import json
import matplotlib.pyplot as plt

from data_util import ModelType, get_human_data, LABEL_FILE

def work_human(attribute: str, **kwargs) -> dict[str, float]:
    with open(LABEL_FILE, 'r') as f:
        desc_translated = json.load(f)
    X, y = get_human_data(attribute, task=None, model_type=ModelType.HUMAN)
    attribute_name = attribute.split('and')[0].strip().lower().replace(' ', '_')
    # count the distribution of y
    distribution = {}
    for ele in desc_translated[attribute_name]:
        if ele == '其他':
            continue
        ele = desc_translated[attribute_name][ele]
        distribution[ele] = 0
    for label in y:
        for ele in label:
            ele = desc_translated[attribute_name][ele]
            distribution[ele] += 1
    # sort the distribution
    distribution = dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    # plot the distribution
    plt.figure(figsize=(10, 7))
    plt.bar(distribution.keys(), distribution.values())
    plt.xticks(list(distribution.keys()), list(distribution.keys()), rotation=25)
    plt.xlabel('Attribute')
    plt.ylabel('Count')
    plt.title(f'Distribution of {attribute}')
    output_file = os.path.join(os.path.dirname(__file__), 'figures', 'human', f'{attribute}.png')
    plt.savefig(output_file)
    return list(distribution.keys())
    
if __name__ == '__main__':
    items = ['Personality', 'Daily Interests and Hobbies', 'Travel Habits', 'Dining Preferences', 'Spending Habits']
    ranks = {}
    for item in items:
        rank = work_human(item)
        ranks[item] = rank
    print(json.dumps(ranks, indent=4))
