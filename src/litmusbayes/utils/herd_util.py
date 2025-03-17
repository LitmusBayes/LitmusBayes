

import os
import sys
from src.litmusbayes import config
from src.litmusbayes.comp.parse_result import parse_one_herd_output, parse_chip_log
from src.litmusbayes.litmus.litmus import LitmusResult, LitmusState
from src.litmusbayes.utils import dir_util, file_util, cmd_util, time_util
from datetime import datetime
from src.litmusbayes.comp.parse_result import parse_herd_log
 
def run_herd(litmus_file_name, litmus_code, cat_file_path) -> LitmusResult:
    """
    Run herd.
    :param litmus_file_name: file name of the litmus test
    :param litmus_code: concrete code of the litmus file
    :param cat_file_path: path of CAT file
    :return: LitmusResult

    herd output example:
    Test 2+2Swap Allowed
    States 4
    0:x10=0; 0:x11=0; 1:x10=1; 1:x11=2; [x]=1; [y]=2;
    0:x10=0; 0:x11=2; 1:x10=0; 1:x11=2; [x]=1; [y]=1;
    0:x10=1; 0:x11=0; 1:x10=1; 1:x11=0; [x]=2; [y]=2;
    0:x10=1; 0:x11=2; 1:x10=0; 1:x11=0; [x]=2; [y]=1;
    Ok
    Witnesses
    Positive: 1 Negative: 3
    Condition exists ([x]=2 /\ [y]=2 /\ 0:x10=1 /\ 0:x11=0 /\ 1:x10=1 /\ 1:x11=0)
    Observation 2+2Swap Sometimes 1 3
    Time 2+2Swap 0.01
    Hash=760e481b3c7b9ad1c78990727e5fcf50
    """
    # TODO: need a more elegant way to set herd output path
    cur_dir = dir_util.get_cur_dir(__file__)
    output_dir = os.path.join(cur_dir, '../../../output', 'litmus_herd_result')
    dir_util.mk_dir_from_dir_path(output_dir)

    # save litmus code
    litmus_file_path = os.path.join(output_dir, litmus_file_name)
    file_util.write_str_to_file(litmus_file_path, litmus_code, False)

    # save herd output
    litmus_herd_result_path = os.path.join(output_dir, litmus_file_name + ".log")
    if cat_file_path is not None:
        cmd = f"{config.HERD7_PATH} {litmus_file_path} -model {cat_file_path}"
    else:
        cmd = f"{config.HERD7_PATH} {litmus_file_path}"
    output = cmd_util.run_cmd_with_output(cmd).strip()
    assert output, f'[ERROR] cmd has empty output: {cmd}'
    # print(f'[INFO] run herd cmd: {cmd}, output: {output}')
    file_util.write_str_to_file(litmus_herd_result_path, output, False)

    states, pos_cnt, neg_cnt, time_cost = parse_one_herd_output(output)
    states = [LitmusState(s) for s in states]
    states.sort(key=lambda s: str(s))
    return LitmusResult(litmus_file_name, states=states, pos_cnt=pos_cnt, neg_cnt=neg_cnt, time_cost=time_cost)

def herd_run_all_valid_litmus(model_file_path, litmus_suite_path, output_path, dependency_dir_path = None):
    output_file = output_path
    file_util.clear_file(output_file)
    index=0 
    start_time = datetime.now()
    
    for litmus_file in litmus_suite_path:
        cmd = f"herd7 {litmus_file} -model {model_file_path}"
        if dependency_dir_path:
            cmd = f"herd7 {litmus_file} -model {model_file_path} -I {dependency_dir_path}"
        cmd = f"{config.HERD_EVAL} {cmd}"
        output = cmd_util.run_cmd_with_output(cmd).strip()
        # print('output',output)
        # print(cmd)
        index+=1
        assert output, f'[ERROR] cmd has empty output: {cmd}'

        time_util.update_start_time()
        file_util.write_str_to_file(output_file, f'[INFO] cmd: {cmd}, time cost: {time_util.cal_time_cost()}\n\n', True)
        file_util.write_str_to_file(output_file, f'{output}\n\n', True)

    file_util.write_str_to_file(output_file,
                                f'[INFO] total time cost: {time_util.get_time_cost(start_time, datetime.now())}\n', True) 



def create_cat_file(index, ppo_list):
    input_cat_def_file = f'{config.INPUT_DIR}/CAT/change/riscv-defs.cat'
    input_cat_file = f'{config.INPUT_DIR}/CAT/change/riscv.cat'
    output_cat_file = f'{config.INPUT_DIR}/CAT/change/riscv-{index}.cat'
    
    with open(input_cat_def_file,'r') as f:
        lines = f.readlines()
    
    with open(input_cat_file,'r') as f:
        cat_lines = f.readlines()

    with open(output_cat_file,'w') as f:
        f.writelines(lines)
        var_str = '\n'
        for i, ppo in enumerate(ppo_list):
            var_str += 'let 'if i==0 else 'and '
            var_str += f'r{i+1} = {ppo} \n'

        ppo_str = '\nlet ppo = \n'
        for i, ppo in enumerate(ppo_list):
            ppo_str += f' ' if i==0 else '| '
            ppo_str += f'r{i+1}\n'
        
        f.write(var_str)
        f.write(ppo_str)
        f.writelines(cat_lines)

    return output_cat_file



    # Tests whether the litmus suite can show differences on the specified ppo
def use_herd_test_ppo_by_litmus(litmus_suite, cat_file_path, output_cat_file):
    input_cat_file = cat_file_path
    print(input_cat_file)
    print(output_cat_file)
    # 1.run herd
    now_log_path_ppo_add = f'{config.INPUT_DIR}/herd/herd_result/after.log'
    now_log_path_ppo_remove = f'{config.INPUT_DIR}/herd/herd_result/before.log'
    herd_run_all_valid_litmus(output_cat_file, litmus_suite, now_log_path_ppo_add, dependency_dir_path = None)    
    herd_run_all_valid_litmus(input_cat_file, litmus_suite, now_log_path_ppo_remove, dependency_dir_path = None)    

    # 2.compare
    now_herd_logs_ppo_add = {r.name: r.states for r in parse_herd_log(now_log_path_ppo_add)}
    now_herd_logs_ppo_remove = {r.name: r.states for r in parse_herd_log(now_log_path_ppo_remove)}
    print('now_herd_logs_ppo_add', now_herd_logs_ppo_add)
    print('now_herd_logs_ppo_remove', now_herd_logs_ppo_remove)
    different_litmus_tests=[]
    for litmus in now_herd_logs_ppo_add:
        if litmus not in now_herd_logs_ppo_remove:
            continue
        if  set(now_herd_logs_ppo_remove[litmus])!=set(now_herd_logs_ppo_add[litmus]):
            print(litmus)
            print(now_herd_logs_ppo_remove[litmus])
            print(now_herd_logs_ppo_add[litmus])
            different_litmus_tests.append(litmus)
    print(different_litmus_tests)  # litmus test path list
    if len(different_litmus_tests) != 0:
        return True
    return False


# get memory model diff litmus test list
def get_model_diff(log_file1, log_file2, mode1 = 'herd', mode2 = 'herd'):

    dif_litmus_list = []
    if mode1 == 'herd':
        model_1_dict = {r.name: r.states for r in parse_herd_log(log_file1)}
    else:
        model_1_dict = {r.name: r.states for r in parse_chip_log(log_file1)}

    if mode2 == 'herd':
        model_2_dict = {r.name: r.states for r in parse_herd_log(log_file2)}
    else:
        model_2_dict = {r.name: r.states for r in parse_chip_log(log_file2)}

    for name in model_1_dict:
        if name in model_2_dict:
            if set(model_1_dict[name]) != set(model_2_dict[name]):
                dif_litmus_list.append(name)
    dif_litmus_list = list(set(dif_litmus_list))
    # print(len(dif_litmus_list))
    # for litmus in dif_litmus_list:
    #     print(litmus)
    return dif_litmus_list







