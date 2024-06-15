from argparse import ArgumentParser
from loguru import logger

from usim import init

class Task:
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def launch(self):
        init()
        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args()
        # log the arguments
        logger.success(args)
        return self.run(**vars(args))