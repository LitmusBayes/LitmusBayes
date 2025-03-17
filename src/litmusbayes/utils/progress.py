
"""Helpers for Graph Plotting"""
import sys
import time


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', start_time=None):
    """Print Progress Bar
    Example:
    >>>    total = 100
    >>>    for i in range(total + 1):
    >>>        print_progress_bar(i, total, prefix='Progress:', suffix='Complete', length=50)
    """
    assert 0 <= iteration <= total
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    output = f'\r{prefix} |{bar}| {percent}%'
    if iteration != total:
        output += f' {suffix}'
        if start_time:
            elapsed_time = "{: .1f}".format(time.time() - start_time)
            output += f' | Elapsed Time: {elapsed_time}s'

    sys.stdout.write(output)
