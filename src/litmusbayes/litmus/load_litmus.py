
import re
from src.litmusbayes.analysis.model.rvwmo import *
from src.litmusbayes.analysis.model.rvwmo_strong_ppo2 import *
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.utils import dir_util, file_util, regex_util
from src.litmusbayes.utils.file_util import *





# cur_dir = dir_util.get_cur_dir(__file__)
# input_dir = os.path.abspath(os.path.join(cur_dir, "../input"))
# litmus_dir = os.path.abspath(os.path.join(input_dir, "./litmus/non-mixed-size"))

input_dir = config.INPUT_DIR
litmus_dir = config.LITMUS_DIR
def get_all_litmus_files() -> List[str]:
    """
    Get all litmus files (size: 3536) after filtering out the exceptional cases.
    """
    print(litmus_dir)
    ori_litmus_files = file_util.list_files(os.path.join(litmus_dir,'non-mixed-size'), '.litmus')
    exceptional_litmus_files = file_util._read_file_to_list_strip(
        os.path.join(input_dir, 'herd/exceptional_aq_rl.txt'))

    all_litmus_files = []
    for litmus_file in ori_litmus_files:
        litmus_name = file_util.get_file_name_from_path(litmus_file)
        if litmus_name not in exceptional_litmus_files:
            all_litmus_files.append(litmus_file)
    return all_litmus_files



def get_litmus_suite_with_specific_ppo(all_litmus_files, ppo):
    """
    A function to explore which tests have the specific ppo (e.g., ppo2)
    As we currently still do not implement the feature: identify the exact ppo rule between two
    events, We have to add code like:
        if 'ppo_r2' in str(ppo_rule):
            pass
    in "rel.py:PPO():wrapper()" function to detect if the specified ppo rule is identified.
    """
    litmus_test_suite = all_litmus_files
    for litmus_file in litmus_test_suite:
        if litmus_file.endswith('/MP+po+ctrl.litmus'):
            continue
        litmus_content = file_util.read_file(litmus_file)
        litmus_instance = parse_litmus(litmus_content)
        rvwmo = RVWMO(plot_enabled=False)
        rvwmo.run(litmus_instance)
    pass


def get_litmus_paths_from_names(all_litmus_files, litmus_names) -> List[str]:
    """
    Get litmus file paths from litmus file names
    """
    litmus_test_suite = []
    for litmus_name in litmus_names:
        if not litmus_name.endswith('.litmus'):
            litmus_name += '.litmus'
        match = False
        for litmus_file in all_litmus_files:
            if litmus_file.endswith(f'/{litmus_name}'):
                litmus_test_suite.append(litmus_file)
                match = True
                break
    return litmus_test_suite

def get_litmus_paths_filter_by_file(all_litmus_files, file_list) -> List[str]:
    litmus_test_suite = []
    filter_suite = []
    for filter_file in file_list:
        with open(filter_file, 'r') as f:
            filter_list = f.readlines()
        filter_list = [file_name.strip() for file_name in filter_list]
        for litmus_name in filter_list:
            if not litmus_name.endswith('.litmus'):
                litmus_name += '.litmus'
                filter_suite.append(litmus_name)
    filter_suite = list(set(filter_suite))
    # for litmus_name in filter_suite:
    #     print(litmus_name)
    for litmus_file in all_litmus_files:
        flag = False
        for filter_file in filter_suite:
            if litmus_file.endswith(f'/{filter_file}'):
                flag = True
                break
        if not flag:
            litmus_test_suite.append(litmus_file)
    return litmus_test_suite

def get_all_litmus_files_sorted_by_time(all_litmus_files):
    """
    parse litmus test info from pytest log file:
    <li class="level test">
        <span><em class="time">
                <div class="time">1.37 s</div>
            </em><em class="status">passed</em>(MP+po+poarp+NEW-state_cnt54)</span>
    </li>
    """
    log_file_string = file_util.read_file(
        os.path.join(input_dir, 'litmus_tests_time_cost.log'))
    time_name_tuples = regex_util.findall(
        '<li class="level test">.*?<div class="time">(.*?)<\/div>.*?\((.*?)-state_.*?\)', log_file_string, re.S)

    # unify the time unit (i.e., s) e.g., 1 m 0 s,210 ms,2.80 s
    def get_time(time):
        elements = time.split(' ')
        if ' m ' in time:
            time = float(elements[0]) * 60 + float(elements[2])
        elif ' ms' in time:
            time = float(elements[0]) / 1000
        else:
            time = float(elements[0])
        return time

    time_name_tuples = [(get_time(time), name.replace('_', '.')) for (time, name) in time_name_tuples]
    time_name_tuples = sorted(time_name_tuples, key=lambda time_name_tuple: time_name_tuple[0])
    total_time_cost = sum([time_name[0] for time_name in time_name_tuples]) / 60  # minutes
    litmus_names = [time_name_tuple[1] for time_name_tuple in time_name_tuples]

    all_litmus_files_sorted_by_time = get_litmus_paths_from_names(all_litmus_files, litmus_names)
    assert len(all_litmus_files_sorted_by_time) == len(litmus_names), 'Some litmus tests are not found in ' \
                                                                      'all_litmus_files.'
    return all_litmus_files_sorted_by_time


class GetLitmusPolicy(Enum):
    All = 0
    SortByTime = 1
    ByName = 2
    ByPPO = 3
    FilterByFile = 4


def get_litmus_by_policy(policy, args):
    litmus_suite = get_all_litmus_files()
    if policy == GetLitmusPolicy.All:
        return litmus_suite
    elif policy == GetLitmusPolicy.SortByTime:
        return get_all_litmus_files_sorted_by_time(litmus_suite)
    elif policy == GetLitmusPolicy.ByName:
        return get_litmus_paths_from_names(litmus_suite, args['name_list'])
    elif policy == GetLitmusPolicy.ByPPO:
        return get_litmus_suite_with_specific_ppo(litmus_suite, args['ppo'])
    elif policy == GetLitmusPolicy.FilterByFile:
        return get_litmus_paths_filter_by_file(litmus_suite, args['file_list'])
    else:
        assert False, 'Unknown policy: {}'.format(policy)

def sort_litmus_by_weight(litmus_name_suite, litmus_dir = [f'{config.INPUT_DIR}/litmus']):
    litmus_content_suite = []
    for litmus in litmus_name_suite:
        for dir in litmus_dir:
            file_path = search_file(litmus, dir, '.litmus')
            if file_path is not None:
                litmus_content_suite.append(parse_litmus(read_file(file_path)))
                break


    weight_list = [litmus.get_cost_by_weight() for litmus in litmus_content_suite]
    weight_litmus_suite = [v for _,v in sorted(zip(weight_list, litmus_name_suite), key = lambda x: x[0])]
    # print(weight_litmus_suite[0:5])
    return weight_litmus_suite