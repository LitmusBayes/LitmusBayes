import os

import numpy as np

from src.litmusbayes.bayes.litmus_params import LitmusParams
from src.litmusbayes.bayes.util import run_litmus_by_mode, parse_log_by_mode, parse_log_by_mode_perple

litmus_repeats_per_iteration = 1 # Number of repeated executions for each litmus test per iteration
max_litmus_repeats_per_iteration = 1
run_time = 100000
litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/bayes_banana_perple'
log_dir_path = '/home/whq/Desktop/code_list/perple_test/bayes_log_banana_perple'

# litmus_dir_path = '/home/software/桌面/bayes/perple_test_riscv/bayes'
# log_dir_path = '/home/software/桌面/bayes/perple_test_riscv/bayes_log'
# litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/qemu'
# log_dir_path = '/home/whq/Desktop/code_list/perple_test/qemu_log'
def run_litmus_and_return_score(litmus_file, config, has_perple, mode = "time", mach = "ssh"):
    log_path = run_litmus_by_mode(litmus_file, config, litmus_dir_path, log_dir_path, run_time = run_time, mode = "exe", mach = mach)

    if has_perple:
        return log_path, parse_log_by_mode_perple(log_path, mode)
    else:
        return log_path, parse_log_by_mode(log_path, mode)


def get_score(litmus_file, config:LitmusParams, mode = "time", mach = "ssh"): # run litmus test

    scores = []
    step = 0
    log_path = None
    while len(scores) < litmus_repeats_per_iteration:
        step += 1
        if step > 10:
            return None, None
            # assert False, f"run litmus test {litmus_file} error"

        for num in range(litmus_repeats_per_iteration):
            # run litmus test
            log_path, score = run_litmus_and_return_score(litmus_file, config, config.is_perple(), mode= mode, mach = mach)
            if score == -1:
                continue
            scores.append(score)

    return log_path, -1

    return log_path, np.median(scores)





# ===========================================================
# Running framework: LitmusRunner
# ===========================================================
class LitmusRunner:
    def __init__(self, litmus_list, config_list, stat_log, mode="time"):
        """
        Initialize the runner.

        Args:
            litmus_list (list): A list of litmus test file names.
            config_list (list): A list of configuration objects for tuning.
            mode (str): Scoring mode, e.g., "time" or "frequency".
        """
        self.litmus_list = litmus_list
        self.config_list = config_list
        self.mode = mode
        self.stat_log = stat_log
        self.has_run_list = {}
        # Task queue: each item is a (litmus_file, config) pair.
        self.tasks = list(zip(self.litmus_list, self.config_list))


        self.index = 0
        # Store results in the form: (litmus_filqe, config, score)
        self.results = []
        self.init_log()
        self.log_f = open(stat_log, "a+")

    def init_log(self):
        if not os.path.exists(self.stat_log):
            with open(self.stat_log, "w") as f:
                pass
        with open(self.stat_log, "r") as f:
            self.has_run_list[f.readline().strip()] = 1

    def record_to_log(self, litmus_file_name):
        self.log_f.write(litmus_file_name)
        self.log_f.write("\n")

    # -----------------------------
    # Get the next task
    # -----------------------------
    def getNext(self):
        """
        Retrieve the next litmus test task.

        Returns:
            tuple or None: (litmus_file, config) if available,
                           None if no tasks remain.
        """
        if self.index >= len(self.tasks):
            return None  # No more tasks

        next_task = self.tasks[self.index]
        self.index += 1
        return next_task

    # -----------------------------
    # Run all tasks
    # -----------------------------
    def run(self):
        """
        Execute all tasks sequentially.

        Returns:
            list: A list of tuples (litmus_file, config, score).
        """
        while True:
            task = self.getNext()
            if task is None:
                break  # Finished all tasks

            lit, cfg = task
            print(lit, cfg)
            log_path, score = get_score(lit, cfg, mode=self.mode)
            print(log_path, score)
            log_path = str(log_path)
            litmus_log_name = ('_').join(log_path.split("/")[-1].split("_")[:-1])
            print(litmus_log_name)
            self.record_to_log(litmus_log_name)
            self.results.append((lit, cfg, score))

        return self.results

    # -----------------------------
    # Statistics summary
    # -----------------------------
    def stats(self):
        """
        Compute summary statistics of all collected results.

        Returns:
            dict: A dictionary containing metrics such as:
                  - count: number of executed tasks
                  - best: task with the lowest score
                  - worst: task with the highest score
                  - median_score: median of scores
                  - mean_score: arithmetic mean of scores
        """
        pass







