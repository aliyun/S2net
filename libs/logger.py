import os, sys, logging, functools
from termcolor import colored

logger_name = "sp_transformer"
logger = logging.getLogger(logger_name)

# We use logger from https://github.com/microsoft/Swin-Transformer/blob/main/logger.py

# Gives you the ability to cache the result of your functions using the Least Recently Used (LRU) strategy.
@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, file_name_prefix=""):

    """ Creating a global logger.

        Creating a global logger to output color formatted logging messages to console terminal in master process.
        And logging messages in all parallel processes will be streamed into log file if specified.

        Args:
            output_dir: output directory.
            dist_rank: rank id in DDP.
            file_name_prefix: log file name prefix. Full logging file name is '{file_name_prefix}_{rank_id}.txt'

        Returns:
            logger: an initialized logger with global "logger_name"
    """

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
    if file_name_prefix != '':
        # create file handlers
        file_handler = logging.FileHandler(os.path.join(output_dir, f'{file_name_prefix}_{dist_rank}.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
    return logger
