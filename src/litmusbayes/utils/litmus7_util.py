import os
import sys

from src.litmusbayes import config
from src.litmusbayes.comp.parse_result import parse_chip_log, parse_herd_log
from src.litmusbayes.litmus.litmus_transformer import get_transform_litmus_list
from src.litmusbayes.utils.file_util import write_line_to_file, rm_files_with_suffix_in_dir
from src.litmusbayes.utils.herd_util import herd_run_all_valid_litmus
from src.litmusbayes.utils.ssh_util import ssh_task

litmus_log7_cnt_dir = {}
litmus_log7_cnt_path = os.path.join(config.LITMUS7_LOG_DIR_PATH, 'litmus7_log_cnt.log')
if not os.path.exists(litmus_log7_cnt_path):
   with open(litmus_log7_cnt_path, 'w') as f:
       f.write('')
else:
    with open(litmus_log7_cnt_path, 'r') as f:
        for line in f:
            litmus_name, cnt = line.strip().split('|')[0],int(line.strip().split('|')[1])
            if litmus_name in litmus_log7_cnt_dir:
                litmus_log7_cnt_dir[litmus_name] = max(litmus_log7_cnt_dir[litmus_name],cnt)
            else:
                litmus_log7_cnt_dir[litmus_name] = cnt


def cmd(cmd):
    return {'type':'run', 'field':cmd}

def put(local_path, remote_path):
    return {'type':'file_in', 'field':f'{local_path},{remote_path}'}

def get(local_path, remote_path):
    return {'type':'file_out', 'field':f'{local_path},{remote_path}'}

def mkdir(path, dir):
    return cmd(f'mkdir -p {path}/{dir}')

def litmus7(path, file_name, dir):
    cmd_litmus7 = f'eval $(opam env); litmus7 -carch RISCV -limit true -mem direct -p 0,1 -barrier userfence -stride 1 -size_of_test 100 -number_of_run 10 -driver C -gcc gcc -ccopts -O2 -linkopt -static -smtmode seq -smt 2  -avail 4 {path}/{file_name}.litmus -o {dir}'
    return cmd(cmd_litmus7)

def make(path):
    return cmd(f'cd {path};make')

def run_litmus(path, log_path, num = 10):
    return cmd(f'{path}/run.exe -r {num} > {log_path}')


def litmus_run(litmus_name, litmus_path, run_time = 10):
    # return None show this litmus test chip don't support
    command = []
    hostname = config.HOSTNAME
    port = config.PORT
    username = config.USERNAME
    password = config.PASSWORD
    host_path = config.HOSTPATH

    host_litmus_dir_name = f'{litmus_name}_{litmus_log7_cnt_dir.get(litmus_name,0)}'
    host_dir = os.path.join(host_path, host_litmus_dir_name)
    host_log_path = os.path.join(host_dir, f'{host_litmus_dir_name}.log')
    local_dir = config.LITMUS7_LOG_DIR_PATH
    local_log_path = os.path.join(local_dir, f'{host_litmus_dir_name}.log')
    command.append(mkdir(host_path, host_litmus_dir_name))
    command.append(put(litmus_path, host_dir))
    litmus_file_name = litmus_path.split('/')[-1].split('.litmus')[0]
    command.append(litmus7(host_dir, litmus_file_name, host_dir))
    command.append(make(host_dir))
    command.append(run_litmus(host_dir, host_log_path, run_time))
    command.append(get(local_dir, host_log_path))
    try_time = 2
    while (True):
        flag, error_list = ssh_task(hostname, port, username, password, command)

        if os.path.exists(local_log_path):
            litmus_log7_cnt_dir.setdefault(litmus_name, 0)
            litmus_log7_cnt_dir[litmus_name] += 1
            write_line_to_file(litmus_log7_cnt_path, f'{litmus_name}|{litmus_log7_cnt_dir[litmus_name]}')
            if os.path.getsize(local_log_path) == 0:
                return None
            break
        else:
            try_time -= 1
        if try_time == 0:
            assert False, 'ssh error'
    result_dict = {r.name : r for r in parse_chip_log(local_log_path)}

    return set(result_dict[litmus_name].states)

def litmus_run_get_log(litmus_name, litmus_path, run_time = 10):
    # return None show this litmus test chip don't support
    command = []
    hostname = config.HOSTNAME
    port = config.PORT
    username = config.USERNAME
    password = config.PASSWORD
    host_path = config.HOSTPATH

    host_litmus_dir_name = f'{litmus_name}_{litmus_log7_cnt_dir.get(litmus_name,0)}'
    host_dir = os.path.join(host_path, host_litmus_dir_name)
    host_log_path = os.path.join(host_dir, f'{host_litmus_dir_name}.log')
    local_dir = config.LITMUS7_LOG_DIR_PATH
    local_log_path = os.path.join(local_dir, f'{host_litmus_dir_name}.log')
    command.append(mkdir(host_path, host_litmus_dir_name))
    command.append(put(litmus_path, host_dir))
    litmus_file_name = litmus_path.split('/')[-1].split('.litmus')[0]
    command.append(litmus7(host_dir, litmus_file_name, host_dir))
    command.append(make(host_dir))
    command.append(run_litmus(host_dir, host_log_path, run_time))
    command.append(get(local_dir, host_log_path))
    try_time = 2
    while (True):
        flag, error_list = ssh_task(hostname, port, username, password, command)

        if os.path.exists(local_log_path):
            litmus_log7_cnt_dir.setdefault(litmus_name, 0)
            litmus_log7_cnt_dir[litmus_name] += 1
            write_line_to_file(litmus_log7_cnt_path, f'{litmus_name}|{litmus_log7_cnt_dir[litmus_name]}')
            if os.path.getsize(local_log_path) == 0:
                return None
            break
        else:
            try_time -= 1
        if try_time == 0:
            assert False, 'ssh error'
    return local_log_path

def litmus_run_until_match(litmus_name, litmus_path, state):
    print(litmus_name)
    assert False, "don't use"
    # run until max time
    run_time = 10
    max_run_time = config.MAX_LITMUS7_TIME

    litmus7_result = None

    while(run_time <= max_run_time):
        result_dict = litmus_run(litmus_name, litmus_path, run_time)
        if result_dict is None:
            return None
        if litmus7_result is None:
            litmus7_result = result_dict
        else:
            litmus7_result.union(result_dict)
        if set(state) == set(litmus7_result):
            return litmus7_result
        run_time *=10
    run_time //=10
    # start transform
    litmus_suite = get_transform_litmus_list(litmus_name, litmus_path, litmus7_result)
    for litmus in litmus_suite:
        result_dict = litmus_run(litmus_name, litmus, run_time)
        if result_dict is None:
            return None
        litmus7_result.union(result_dict)
        if set(state) == set(litmus7_result):
            return litmus7_result

    print(set(litmus7_result))
    return litmus7_result


def integrate_chip_log(log_path):
    with open(log_path, 'a+') as f:
        for root, _, files in os.walk(config.LITMUS7_LOG_DIR_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                if not file_path.endswith('.log'):
                    continue
                if file_path == litmus_log7_cnt_path:
                    continue
                with open(file_path, 'r') as rf:
                    f.write(rf.read())
                    f.write('\n')
    rm_files_with_suffix_in_dir(config.LITMUS7_LOG_DIR_PATH, '.log')


if __name__ == '__main__':
    # litmus_name = 'SB+rfi-data-rfis'
    # litmus_path = os.path.join(config.LITMUS_DIR,'non-mixed-size/RELAX/Rfi/SB+rfi-data-rfis.litmus')
    # litmus_name = 'LB'
    # litmus_path = os.path.join(config.LITMUS_TRANS_DIR_PATH,'LB_3.litmus')
    host_name = config.HOSTNAME
    port = config.PORT
    username = config.USERNAME
    password = config.PASSWORD
    host_path = config.HOSTPATH
    # state_dict = {r.name :r.states for r in parse_herd_log(os.path.join(config.CAT_DIR,'../herd/herd_results_rvwmo.log'))}
    # state = state_dict[litmus_name]
    # # litmus_run(litmus_name, litmus_path, host_name, port, username, password, host_path, run_time = 10)
    # litmus_run_until_match(litmus_name, litmus_path, host_name, port, username, password, host_path, state)
    # chip_dict = {r.name :r.states for r in parse_chip_log(os.path.join(config.INPUT_DIR,'chip_execution_logs/C910/chip_log.txt'))}
    # chip_state = chip_dict[litmus_name]
    # get_transform_litmus_list(litmus_name, litmus_path, chip_state)

    # integrate_chip_log()
    # integrate_chip_log(os.path.join(config.TEST_DIR,'input/chip_execution_logs/C910/chip_log.txt'))


    #  test suite
    litmus_suite = []
    litmus_dir = os.path.join(config.OUTPUT_DIR, 'complete_litmus')
    for root, _, files in os.walk(litmus_dir):
        for file in files:
            if file.endswith('.litmus'):
                litmus_suite.append(os.path.join(root, file))
    path = os.path.join(config.OUTPUT_DIR, 'complete_litmus/herd_complete_litmus.log')
    cat_path =os.path.join(config.CAT_DIR, 'riscv-complete.cat')
    herd_run_all_valid_litmus(cat_path, litmus_suite, path)
    state_dict = {r.name :r.states for r in parse_herd_log(path)}

    for litmus in litmus_suite:
        litmus_name = litmus.split('/')[-1].split('.litmus')[0]
        state = state_dict[litmus_name]
        print(state)
        litmus_run(litmus_name, litmus, host_name, port, username, password, host_path, run_time = 10)
        # litmus_run_until_match(litmus_name, litmus, host_name, port, username, password, host_path, state)
    integrate_chip_log(os.path.join(config.TEST_DIR,'input/chip_execution_logs/C910/chip_log.txt'))
    # chip_state_dict = {r.name: r.states for r in parse_chip_log(os.path.join(config.TEST_DIR,'input/chip_execution_logs/C910/chip_log.txt'))}
    # with open('suc_log.log', 'w') as f:
    #     od = sys.stdout
    #     sys.stdout = f
    #     for litmus in litmus_suite:
    #         litmus_name = litmus.split('/')[-1].split('.litmus')[0]
    #         chip_state = chip_state_dict[litmus_name]
    #         herd_state = state_dict[litmus_name]
    #         if set(chip_state) != set(herd_state):
    #             print(litmus)
    #             print(chip_state)
    #             print(herd_state)
    #     sys.stdout = od