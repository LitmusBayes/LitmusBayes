from src.litmusbayes.bayes.framework import LitmusRunner, get_score
from src.litmusbayes.bayes.litmus_params import LitmusParams
from src.litmusbayes.bayes.util import get_files, read_log_to_summary



class LitmusRunnerForQemu(LitmusRunner):
    def __init__(self, litmus_list, config_list, stat_log, mode="time"):
        super().__init__(litmus_list, config_list, stat_log, mode)


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
            log_path, score = get_score(lit, cfg, mode=self.mode, mach="qemu")
            litmus_log_name = ('_').join(log_path.split("/")[-1].split("_")[:-1])
            print(litmus_log_name)
            self.record_to_log(litmus_log_name)
            self.results.append((lit, cfg, score))

        return self.results




if __name__ == "__main__":
    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple"
    stat_log = "/home/whq/Desktop/code_list/perple_test/qemu_stat/log_record.log"
    dir_path = "/home/whq/Desktop/code_list/perple_test/qemu_log"
    log_path = "/home/whq/Desktop/code_list/perple_test/qemu_stat/log_stat.csv"

    litmus_list = get_files(litmus_path)
    for litmus in litmus_list:
        print(litmus)

    config_list = [LitmusParams() for _ in litmus_list]
    vector = [-1,5,-1,-1,-1,-1,-1,-1,-1,-1,1]
    for item in config_list:
        item.from_vector(vector)
        item.apply_standard_form()

    runner = LitmusRunnerForQemu(litmus_list, config_list, stat_log)
    runner.run()

    read_log_to_summary(dir_path, log_path)