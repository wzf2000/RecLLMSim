import os
import json
import pandas as pd

from utils import SIM_DIR, HUMAN_DIR
from tag import read_labels

def get_sim_profile(dir_name: str = SIM_DIR) -> pd.DataFrame:
    task = 'travel planning'
    files = os.listdir(os.path.join(dir_name, task))
    files = [file for file in files if file.endswith('.json')]
    profiles = set()
    profile_list = []
    for file in files:
        with open(os.path.join(dir_name, task, file), 'r') as f:
            data = json.load(f)
        profile = data['preference']
        if profile not in profiles:
            profiles.add(data['preference'])
            profile_list.append(data['profile_mapped'])
    data_group: dict[str, list] = {}
    for attribute in profile_list[0]:
        attribute_key = attribute.split('and')[0].strip().lower().replace(' ', '_')
        data_group[attribute_key] = []
    for profile in profile_list:
        for attribute in profile:
            attribute_key = attribute.split('and')[0].strip().lower().replace(' ', '_')
            if isinstance(profile[attribute], dict):
                data_group[attribute_key].append(profile[attribute]['en'])
            elif isinstance(profile[attribute], list):
                data_group[attribute_key].append([item['en'] for item in profile[attribute]])
    return pd.DataFrame(data_group)

def get_human_profile(dir_name: str = HUMAN_DIR) -> pd.DataFrame:
    labels = read_labels()
    profile_list = []
    for user in os.listdir(dir_name):
        tasks = os.listdir(os.path.join(dir_name, user))
        tasks = [task for task in tasks if os.path.isdir(os.path.join(dir_name, user, task))]
        task = tasks[0]
        files = os.listdir(os.path.join(dir_name, user, task))
        files = [file for file in files if file.endswith('.json')]
        file = files[0]
        with open(os.path.join(dir_name, user, task, file), 'r') as f:
            data = json.load(f)
            profile_list.append(data['profile'])
    data_group: dict[str, list] = {}
    for attribute in profile_list[0]:
        data_group[attribute] = []
    for profile in profile_list:
        for attribute in profile:
            if isinstance(profile[attribute], (str, int)):
                if attribute in labels:
                    data_group[attribute].append(labels[attribute][profile[attribute]])
                else:
                    data_group[attribute].append(profile[attribute])
            elif isinstance(profile[attribute], list):
                data_group[attribute].append([labels[attribute][item] for item in profile[attribute]])
    return pd.DataFrame(data_group)

if __name__ == '__main__':
    df = get_human_profile()
    print(df.head())
    df = get_sim_profile()
    print(df.head())
