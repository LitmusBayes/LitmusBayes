from src.litmusbayes.bayes.framework import LitmusRunner, get_score
from src.litmusbayes.bayes.litmus_params import LitmusParams
from src.litmusbayes.bayes.util import get_files




litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple"
stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record1.log"

if __name__ == "__main__":
    # litmus_list = get_litmus_files(litmus_path)
    # for litmus in litmus_list:
    #     print(litmus)

    litmus_list = ['/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/SB+fence.i+fence.rw.rw.litmus']

    config_list = [LitmusParams({'a':2,'smt':1}) for _ in litmus_list]
    vector = [-1,5,-1,-1,-1,-1,-1,-1,-1,-1,1]
    for item in config_list:
        item.from_vector(vector)
        item.apply_standard_form()

    # config_list = [item.apply_standard_form() for item in config_list]

    # config_list = [item.from_vector(vector) for item in config_list]
    # for path in litmus_list:
    #     print(path)
    # runner = LitmusRunner(litmus_list, config_list, stat_log)
    # runner.run()
    get_score(litmus_list[0], config_list[0], True)

