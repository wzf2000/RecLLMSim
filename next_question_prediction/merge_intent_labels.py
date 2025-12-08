import os
import sys
import json
from tqdm import tqdm
from loguru import logger

from nqp_data import get_nqp_data, get_nqp_data_sim, get_task_list

def main(human: bool = True):
    if human:
        logger.info("Merging intent labels for human data...")
        train_data, test_data = get_nqp_data(sample=False, task_list=get_task_list(human=True, task_name='all'))
        data = train_data + test_data
    else:
        logger.info("Merging intent labels for simulation data...")
        data = get_nqp_data_sim(task_list=get_task_list(human=False, task_name='all'))
    label_dir = './user_intent_classification_results' if human else './user_intent_classification_results_sim'
    for i, item in tqdm(enumerate(data)):
        label_file = os.path.join(label_dir, f'results_{i}.json')
        with open(label_file, 'r') as f:
            label_data = json.load(f)
        assert label_data['file_path'] == item['file_path']
        assert label_data['task'] == item['task']
        with open(item['file_path'], 'r', encoding='utf-8') as f:
            dialog_data = json.load(f)
        assert dialog_data['history'][item['turn']]['role'] == 'user'
        if 'intent_label' in dialog_data['history'][item['turn']] and 'intent_reason' in dialog_data['history'][item['turn']]:
            logger.warning(f"File {item['file_path']} turn {item['turn']} already has intent label, skipping.")
            continue
        dialog_data['history'][item['turn']]['intent_label'] = label_data['predicted_label']
        dialog_data['history'][item['turn']]['intent_reason'] = label_data['reason']
        with open(item['file_path'], 'w', encoding='utf-8') as f:
            json.dump(dialog_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0 and 'sim' in args:
        main(human=False)
    else:
        main()
