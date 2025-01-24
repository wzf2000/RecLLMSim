import os

SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLM_agent_user')
HUMAN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real_human_user')
LABEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_profiling', 'desc_translated.json')
