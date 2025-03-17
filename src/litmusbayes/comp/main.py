


"""
this file aims to parse the chip logs and output info:
1)basic information
2)whether the results conform to cx.cat
3)whether the results conform to rvwmo.cat
4)the coverage of existing test results
"""

import os

from src.litmusbayes import config
from src.litmusbayes.comp.parse_result import parse_chip_log, parse_herd_log
from src.litmusbayes.utils import dir_util, file_util, list_util, dataframe_util, seaborn_util, regex_util

cur_dir = dir_util.get_cur_dir(__file__)
input_dir = os.path.abspath(os.path.join(config.TEST_DIR, "input"))


def get_num_threads(litmus_code: str):
    results = regex_util.findall(r"P(.*?)", litmus_code)
    return len(results)


def parse_uniq_states_in_chip_cnt(all_litmus_chip_results, model_name, print_list=True):
    print(f"============Comparison with {model_name} model============")
    herd_log_path = os.path.join(input_dir, f'herd/herd_results_{model_name}.log')

    uniq_states_in_chip_cnt = 0
    unique_info_pairs = []
    coverage_df = dataframe_util.init_df(
        ["litmus_name", "coverage_rate", "coverage_rate_str", "actual_states", "ideal_states", "num_threads"])

    litmus_herd_results = parse_herd_log(herd_log_path)
    for litmus_chip_result in all_litmus_chip_results:
        litmus_herd_result = litmus_herd_results[litmus_herd_results.index(litmus_chip_result)]
        uniq_states = list_util.get_uniq_in_src(litmus_chip_result.states, litmus_herd_result.states)
        num_threads = get_num_threads(litmus_chip_result.litmus_code)

        if len(uniq_states) > 0:
            uniq_states_in_chip_cnt += 1
            unique_info_pairs.append((litmus_chip_result.name, len(uniq_states), num_threads))

        # assert len(litmus_chip_result.states) <= len(litmus_herd_result.states)
        coverage_rate = len(litmus_chip_result.states) / len(litmus_herd_result.states)
        coverage_df.loc[coverage_df.shape[0]] = [litmus_chip_result.name,
                                                 round(coverage_rate, 4),
                                                 f"{round(coverage_rate * 100, 2)}%",
                                                 len(litmus_chip_result.states),
                                                 len(litmus_herd_result.states),
                                                 num_threads
                                                 ]

    print(
        f"The results conform to {model_name}.cat? {uniq_states_in_chip_cnt == 0} (uniq_states_in_chip_cnt: {uniq_states_in_chip_cnt})")
    if uniq_states_in_chip_cnt > 0 and print_list:
        print("[List]: Litmus Tests That Violate The Memory Model")
        print("ID \t litmus_name             \t num_unique_states \t num_threads")
        unique_info_pairs.sort(key=lambda pair: pair[1])
        for i, unique_info_pair in enumerate(unique_info_pairs):
            assert unique_info_pair[2] > 2  # litmus test that violates rvwmo model must have more than 2 threads
            print(f"{i} \t {unique_info_pair[0]:<30} \t {unique_info_pair[1]} \t {unique_info_pair[2]}")

    # the coverage of existing test results
    print(
        f"num_litmus_tests: {coverage_df.shape[0]}, num_litmus_tests(coverage rate < 1): {coverage_df[coverage_df['coverage_rate'] < 1].shape[0]}, num_litmus_tests(coverage rate == 1): {coverage_df[coverage_df['coverage_rate'] == 1].shape[0]}")
    filtered_coverage_df = coverage_df[coverage_df['coverage_rate'] != 1].copy(deep=True)
    filtered_coverage_df.sort_values("coverage_rate", inplace=True, ascending=True, ignore_index=True)

    fig_output_file = os.path.join(config.OUTPUT_DIR, f'{model_name}_coverage_rate')
    df_output_file = os.path.join(config.OUTPUT_DIR, f'{model_name}_coverage_rate.df')
    seaborn_util.draw_single_boxplot(filtered_coverage_df, (6, 8), 'coverage_rate', "", "coverage",
                                     fig_output_file)
    dataframe_util.write_csv(df_output_file,
                             filtered_coverage_df[
                                 ['litmus_name', 'coverage_rate_str', "actual_states", "ideal_states", "num_threads"]])

    return coverage_df


def parse_all_logs_a_time(chip_log_paths):
    """
    parse all logs a time (i.e., merge all logs and parse)
    """
    all_litmus_chip_results = []
    for i, log_path in enumerate(chip_log_paths):
        litmus_chip_results = parse_chip_log(log_path)
        # basic information
        if i == 0:
            print("============BASIC INFO============\nlog_path_name \t num_litmus_tests \t num_trials_each_test")
        print(
            f"{file_util.get_file_name_from_path(log_path, True)} \t {len(litmus_chip_results)}  \t {litmus_chip_results[0].pos_cnt + litmus_chip_results[0].neg_cnt}")
        assert not list_util.has_duplicates(litmus_chip_results)
        for litmus_chip_result in litmus_chip_results:
            if litmus_chip_result not in all_litmus_chip_results:
                all_litmus_chip_results.append(litmus_chip_result)
            else:
                all_litmus_chip_results[all_litmus_chip_results.index(litmus_chip_result)].union(litmus_chip_result)

    # whether the results conform to cx.cat and rvwmo.cat
    parse_uniq_states_in_chip_cnt(all_litmus_chip_results, 'cx')
    parse_uniq_states_in_chip_cnt(all_litmus_chip_results, 'rvwmo')


def parse_one_log_a_time(chip_log_paths):
    """
    parse one log a time
    """
    cx_logs_df = dataframe_util.init_df(
        ["log_name", "num_trials_each_test", "num_litmus", "num_litmus_cov_rate_le_1", "actual_states",
         "percentage_cov_le_1"])
    for i, log_path in enumerate(chip_log_paths):
        litmus_chip_results = parse_chip_log(log_path)
        # basic information
        log_path_name = file_util.get_file_name_from_path(log_path, True)
        num_trials_each_test = litmus_chip_results[0].pos_cnt + litmus_chip_results[0].neg_cnt
        print("\n============BASIC INFO============\nlog_path_name \t num_litmus_tests \t num_trials_each_test")
        print(f"{log_path_name} \t {len(litmus_chip_results)}  \t {num_trials_each_test}")
        assert not list_util.has_duplicates(litmus_chip_results)

        # whether the results conform to cx.cat and rvwmo.cat
        cur_cx_df = parse_uniq_states_in_chip_cnt(litmus_chip_results, 'cx', False)
        cx_logs_df.loc[cx_logs_df.shape[0]] = [log_path_name,
                                               num_trials_each_test,
                                               cur_cx_df.shape[0],
                                               cur_cx_df[cur_cx_df['coverage_rate'] < 1].shape[0],
                                               cur_cx_df['actual_states'].sum(),
                                               cur_cx_df[cur_cx_df['coverage_rate'] < 1].shape[0] / cur_cx_df.shape[0]
                                               ]

        parse_uniq_states_in_chip_cnt(litmus_chip_results, 'rvwmo', print_list=True)

    df_output_file = os.path.join(config.OUTPUT_DIR, f'cx_logs_coverage.df')
    cx_logs_df.sort_values("percentage_cov_le_1", inplace=True, ascending=True, ignore_index=True)
    dataframe_util.write_df_str(df_output_file, cx_logs_df)


if __name__ == "__main__":
    # manually specified
    chip_log_paths = [f'{input_dir}/chip_execution_logs/LITMUS-v1.3.106-20240412/run.{x}.log' for x in range(1, 6)] + [
        f'{input_dir}/chip_execution_logs/LITMUS-v1.3.106-20240412/run.test.log']
    # + [f'{input_dir}/chip_execution_logs/cx1c-20240415/run.1.log.20240410']
    # chip_log_paths = [f'{input_dir}/chip_execution_logs/cx1c-20240415/run.1.log.20240410']

    parse_all_logs_a_time(chip_log_paths)
    # parse_one_log_a_time(chip_log_paths)
