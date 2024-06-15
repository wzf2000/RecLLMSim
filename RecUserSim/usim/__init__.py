import os
from loguru import logger

CHAT_VERSION = '4' if 'CHAT_VERSION' not in os.environ else os.environ['CHAT_VERSION']

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'output')
OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, 'data')
OUTPUT_EXP_PATH = os.path.join(OUTPUT_PATH, 'human_exp')
CONFIG_PATH = os.path.join(ROOT_PATH, 'config')
POOL_PATH = os.path.join(DATA_PATH, 'pool')
CONFIG_FILE = os.path.join(CONFIG_PATH, f'configV{CHAT_VERSION}.json')
WEB_USER_CONFIG_FILE = os.path.join(CONFIG_PATH, 'web_user.yaml')

POOL_NAME = 'preference_poolV{}.txt'

PREFERENCE = '[PREFERENCE]'
TASK_CONTEXT = '[TASK_CONTEXT]'
POOL_SIZE = '[POOL_SIZE]'

def init(prefix: str = ''):
    # Add log file, with timestamp as file name
    logger.add(f"log/{prefix + '_' if prefix != '' else ''}chatV{CHAT_VERSION}" + "_{time:YYYY-MM-DD:HH:mm:ss}.log", rotation="500 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")