from argparse import ArgumentParser

from usim.tasks.base_task import Task
from usim.utils.preferences import build_new_preference_pool

class PGTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser):
        parser.add_argument('-c', '--config', default='config/preference_gen_en.json', type=str, help='The path of config file')
        parser.add_argument('-p', '--pool_size', default=50, type=int, help='The size of preference pool')
        return parser
    
    def run(self, config: str, pool_size: int, *args, **kwargs):
        build_new_preference_pool(config, pool_size)
