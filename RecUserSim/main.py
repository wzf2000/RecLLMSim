import sys
from argparse import ArgumentParser
from loguru import logger
from usim.tasks import PGTask, GenChatTask

def main():
    init_parser = ArgumentParser()
    init_parser.add_argument('-m', '--mode', type=str, default='preference', help='The main function to run')
    init_parser.add_argument('--verbose', type=str, default='INFO', choices=['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'], help='The log level')
    init_args, init_extras = init_parser.parse_known_args()
    logger.remove()
    logger.add(sys.stderr, level=init_args.verbose)
    if init_args.mode == 'preference':
        task = PGTask()
    elif init_args.mode == 'gen_chat':
        task = GenChatTask()
    else:
        logger.warning('No such mode!')
    task.launch()

if __name__ == '__main__':
    main()
