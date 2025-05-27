import os

SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLM_agent_user')
HUMAN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real_human_user')
HUMAN_DIR_V2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'human_exp_V2')
LABEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_profiling', 'desc_translated.json')
