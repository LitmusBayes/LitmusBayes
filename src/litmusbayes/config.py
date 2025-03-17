

"""Global Variable Dictionary"""
import os

from src.litmusbayes.utils import dir_util
import platform

REG_SIZE = 64

READS_PERMUTATION_REDUCTION = True
def init():
    """Initialize the global variable dictionary."""
    global _global_dict
    _global_dict = {}


def reset():
    """Reset the global variable dictionary."""
    _global_dict.clear()


def set_var(key: str, value):
    """Set the global variable `key` to `value`.
    as set() will conflict with the following set() method (list to set). So I change the name from set to set_val

    Parameters
    ----------
    key : str
        The name of the variable to set.
    value
        The value to be set.
    """
    _global_dict[key] = value


def get_var(key: str):
    """It returns the value of the global variable `key`.

    Parameters
    ----------
    key : str
        The name of the variable.

    Returns
    -------
    Any
        The value of the global variable `key`.
    """
    try:
        return _global_dict[key]
    except KeyError:
        print(f"fail to read {key}")


def has(key: str):
    """True if `key` in global variables, False otherwise.

    Parameters
    ----------
    key: str
        The variable name.

    Returns
    -------
    bool
        True if `key` in global variables, False otherwise.
    """
    return key in _global_dict.keys()


CUR_DIR = dir_util.get_cur_dir(__file__)

# constant: tool dir/paths
TOOL_DIR = os.path.join(CUR_DIR, "../../tools")
GLIBC_VERSION = platform.platform()[-9:]
# HERD7_PATH = shutil.which("herd7")
# assert HERD7_PATH is not None
# HERD7_PATH = os.path.join("/home/xuezheng/.local/opt/herdtools7/bin//", "herd7")
HERD7_PATH = os.path.join(TOOL_DIR, 'herdtools', GLIBC_VERSION, 'herd7')
DIYONE7_PATH = os.path.join(TOOL_DIR, 'herdtools', 'diyone7')

#
OUTPUT_DIR = os.path.join(CUR_DIR, "../output")
dir_util.mk_dir_from_dir_path(OUTPUT_DIR)
TEST_DIR = os.path.join(CUR_DIR, "../../tests")

#
INPUT_DIR = os.path.join(CUR_DIR, "../../tests/input")
CAT_DIR = os.path.join(CUR_DIR,'../../tests/input/CAT')
LITMUS_DIR = os.path.join(CUR_DIR, '../../tests/input/litmus')

DIY7_CACHE_FILE_PATH = os.path.join(CUR_DIR, '../../tests/diy_cache.txt')
HERD_EVAL = 'eval $(opam env);'

LITMUS7_LOG_DIR_PATH = os.path.join(CUR_DIR, '../../log/litmus7')
LITMUS_TRANS_DIR_PATH = os.path.join(CUR_DIR, '../../log/litmus')
HERD_LOG_DIR_PATH = os.path.join(CUR_DIR, '../../log/herd')

MAX_LITMUS7_TIME = 100000
LOG_PATH = os.path.join(CUR_DIR, '../../tests/log.txt')

C910_log_path = os.path.join(TEST_DIR,'input/chip_execution_logs/C910/chip_log.txt')

HOSTNAME = '10.42.0.1'
PORT = 22
USERNAME = 'sipeed'
PASSWORD = 'sipeed'
HOSTPATH = '/home/sipeed/riscv'
RELATIONS = [
            'rmw',
            'addr',
            'data',
            'ctrl',
            'fence',
            'po',
            'loc',
            'rf',
            'rfi',
            'rfe',
            'rsw',
            'co',
            'coi',
            'coe',
            'fr',
            'ppo',
        ]