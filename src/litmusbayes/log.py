

"""Logging"""

import pprint
import sys

from loguru import logger

from src.litmusbayes import config

LOG_LEVELS = [
    'DEBUG',
    'INFO',
    'WARNING',
    'ERROR'
]

LOG_LEVEL = 'INFO'

# set the debug level here
logger.remove()
logger.add(config.LOG_PATH, level="DEBUG")
logger.add(sys.stderr, level="ERROR")


def level_checker(func):
    # a decorator for checking the log level
    def wrapper(*args, **kwargs):
        if LOG_LEVELS.index(LOG_LEVEL) <= LOG_LEVELS.index(func.__name__):
            func(*args, **kwargs)

    return wrapper


def _format(obj, comments):
    return ('\n' + (comments + '\n') if comments else '') + pprint.pformat(obj)


@level_checker
def DEBUG(obj, comments=''):
    logger.opt(depth=1).debug(_format(obj, comments))


@level_checker
def INFO(obj, comments=''):
    logger.opt(depth=1).info(_format(obj, comments))


@level_checker
def WARNING(obj, comments=''):
    logger.opt(depth=1).warning(_format(obj, comments))


@level_checker
def ERROR(obj, comments=''):
    logger.opt(depth=1).error(_format(obj, comments))
