import os
import re
from codecs import strict_errors

from src.litmusbayes.comp.parse_result import parse_chip_log
from src.litmusbayes.litmus.litmus_transformer import litmus_name
from src.litmusbayes.litmus.load_litmus import input_dir
from src.litmusbayes.perple.trans_cpp import filter_Thread
from src.litmusbayes.utils.cmd_util import run_cmd
import paramiko

def build_cmd(config: dict) -> str:
    # 默认参数
    defaults = {
        "out_dir": "",
        "arch": "RISCV",
        "limit": True,
        "mem": "direct",
        "p": "0,1",
        "barrier": "userfence",
        'delay': 1024,
        "stride": 31,
        "size_of_test": 100,
        "number_of_run": 10,
        "driver": "C",
        "gcc": "riscv64-unknown-linux-gnu-gcc",
        "ccopts": "-O2",
        "linkopt": "-static",
        "smtmode": "seq",
        "smt": 1,
        "avail": 2,
        "litmus_file": "",
        'noccs': 1,
        'cautious': 'false',
        'alloc': 'dynamic',
        'doublealloc': 'false',
        'mode': 'std'
    }

    # 用户配置覆盖默认值
    params = {**defaults, **config}

    cmd = [
        f"mkdir -p {params['out_dir']}; ",
        "eval $(opam env); ",
        "litmus7",
        f"-carch {params['arch']}",
        f"-limit {'true' if params['limit'] else 'false'}",
        f"-mem {params['mem']}",
        f"-p {params['p']}",
        f"-barrier {params['barrier']}",
        f"-stride {params['stride']}",
        f"-size_of_test {params['size_of_test']}",
        f"-number_of_run {params['number_of_run']}",
        f"-driver {params['driver']}",
        f"-gcc {params['gcc']}",
        f"-ccopts {params['ccopts']}",
        f"-linkopt {params['linkopt']}",
        f"-smtmode {params['smtmode']}",
        f"-smt {params['smt']}",
        f"-avail {params['avail']}",
        f"-cautious {params['cautious']}",
        f"-alloc {params['alloc']}",
        f"-doublealloc {params['doublealloc']}",
        f"-delay {params['delay']}",
        params['litmus_file'],
        f"-o {params['out_dir']}"
    ]
    return " ".join(cmd)

def build_none_cmd(out_dir, litmus_file):
    run_cmd(build_cmd({'out_dir': out_dir, 'litmus_file': litmus_file, 'barrier': 'none'}))
    run_cmd(f'cd {out_dir};make')

def build_arg_cmd(out_dir, litmus_file):
    arg = {
        'out_dir': out_dir,
        'litmus_file': litmus_file,
        'barrier': 'none',
        'delay' : 1024,
        'mem': 'direct',
        'noccs': 2,
        'cautious': 'false',
        'alloc': 'before',
        'doublealloc': 'false',
        'mode': 'presi'

    }
    run_cmd(build_cmd(arg))
    run_cmd(f'cd {out_dir};make')


def build_userfence_cmd(out_dir, litmus_file):
    if os.path.exists(f'{out_dir}/run.exe'):
        return
    run_cmd(build_cmd({'out_dir': out_dir, 'litmus_file': litmus_file, 'barrier': 'userfence'}))
    print(build_cmd({'out_dir': out_dir, 'litmus_file': litmus_file, 'barrier': 'userfence'}))

    run_cmd(f'cd {out_dir};make')

def build_perple_cmd(out_dir, litmus_file):
    print(build_cmd({'out_dir': out_dir, 'litmus_file': litmus_file, 'barrier': 'none'}))
    run_cmd(build_cmd({'out_dir': out_dir, 'litmus_file': litmus_file, 'barrier': 'none'}))
    litmus_name = out_dir.split("/")[-1]
    input_path = os.path.join(out_dir, f'{litmus_name}.c')
    filter_Thread(input_path, input_path, litmus_file, mode = 'h')
    run_cmd(f'cd {out_dir};make')



def run_ssh(input_dir, chip_dir, litmus_name, mode = "", time = 0, run_time = 100):
    import paramiko

    # 配置 SSH 登录信息
    host = "192.168.201.168"  # 远程服务器地址
    port = 22  # SSH 端口
    username = "sipeed"  # SSH 用户名
    password = "sipeed"  # SSH 密码

    # 文件路径
    local_file = os.path.join(input_dir, f"run.exe")  # 本地要上传的文件
    remote_file = os.path.join(chip_dir, "run.exe")  # 上传到远程的路径
    remote_log = os.path.join(chip_dir, f"{litmus_name}_{mode}_{run_time}.log")  # 远程日志文件
    local_log = os.path.join(input_dir, f"{litmus_name}_{mode}_{run_time}-{time}.log")  # 下载回本地的路径

    # print(local_file)
    # print(remote_file)
    # print(remote_log)
    # print(local_log)

    # 创建 SSH 客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接远程服务器
    ssh.connect(host, port, username, password)

    # 打开 SFTP 传输
    sftp = ssh.open_sftp()

    # 上传文件
    print(f"Uploading {local_file} -> {remote_file}")
    sftp.put(local_file, remote_file)

    # 修改权限，确保可执行
    ssh.exec_command(f"chmod +x {remote_file}")

    # 运行命令并输出到日志文件
    print("Running remote command...")
    stdin, stdout, stderr = ssh.exec_command(f"{remote_file} -s {run_time} > {remote_log} 2>&1")

    # 阻塞等待远程命令执行完成
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print("Remote command finished successfully")
    else:
        print(f"Remote command failed with exit status {exit_status}")

    # 下载日志文件
    print(f"Downloading {remote_log} -> {local_log}")
    sftp.get(remote_log, local_log)

    # 关闭连接
    sftp.close()
    ssh.close()

    print("Done.")


def run_perple_in_chip(input_dir, chip_dir, litmus_name):
    run_ssh(input_dir, chip_dir, litmus_name, mode='perple', run_time=100000)

def run_userfence_in_chip(input_dir, chip_dir, litmus_name):
    run_ssh(input_dir, chip_dir, litmus_name, mode='userfence', run_time=100000)

def run_none_in_chip(input_dir, chip_dir, litmus_name):
    run_ssh(input_dir, chip_dir, litmus_name, mode='none', run_time=100000)

def run_arg_in_chip(input_dir, chip_dir, litmus_name, mode, time):
    run_ssh(input_dir, chip_dir, litmus_name, mode= mode, time = time, run_time=100000)

test_dir = '/home/whq/Desktop/code_list/perple_test'
# litmus_dir = os.path.join(test_dir, 'litmus_test')
litmus_dir = os.path.join(test_dir, 'litmus_test')
perple_dir = os.path.join(test_dir, 'perple')
userfence_dir = os.path.join(test_dir, 'delay_userfence')
none_dir = os.path.join(test_dir, 'none')
arg_dir = os.path.join(test_dir, 'stress/arg_perple')

def statistic_state(statistic_dir, output_dir, mean_flag = False):
    log_files = []
    for dirpath, _, filenames in os.walk(statistic_dir):
        for filename in filenames:
            if filename.endswith(".log"):
                os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, filename.replace("100000+", "100000-")))
                log_files.append(os.path.join(dirpath, filename))

    log = {}
    mean_log = {}
    for log_file in log_files:
        litmus_name = log_file.split("/")[-1].split(".")[0]
        litmus_name_without_suffix = litmus_name.split("_")[0]

        if litmus_name not in log:
            log[litmus_name] = {}

        state_dict = parse_chip_log(log_file)

        state_dict = {r.name: (r.states, r.time_cost) for r in state_dict}
        print(state_dict)
        if litmus_name_without_suffix in state_dict:
            state_dict = state_dict[litmus_name_without_suffix]
            if litmus_name_without_suffix not in log:
                log[litmus_name_without_suffix] = {}
            if litmus_name_without_suffix not in mean_log:
                mean_log[litmus_name_without_suffix] = {}
            print(state_dict)
            if len(state_dict) == 2:
                states, timecost = state_dict[0], state_dict[1]
                for state in states:
                    state_num = state.num
                    if mean_flag:
                        if str(state) not in mean_log[litmus_name_without_suffix]:
                            mean_log[litmus_name_without_suffix][str(state)] = {}
                        litmus_name_without_time = litmus_name.split("-")[0]
                        if litmus_name_without_time not in mean_log[litmus_name_without_suffix][str(state)]:
                            mean_log[litmus_name_without_suffix][str(state)][litmus_name_without_time]=(1, state_num, timecost)
                        else:
                            mean_log[litmus_name_without_suffix][str(state)][litmus_name_without_time] = (mean_log[litmus_name_without_suffix][str(state)][litmus_name_without_time][0]+1, mean_log[litmus_name_without_suffix][str(state)][litmus_name_without_time][1]+state_num, mean_log[litmus_name_without_suffix][str(state)][litmus_name_without_time][2]+timecost)
                    else:
                        if str(state) not in log[litmus_name_without_suffix]:
                            log[litmus_name_without_suffix][str(state)] = []
                        log[litmus_name_without_suffix][str(state)].append((litmus_name, state_num, state_num/timecost))
                if mean_flag:
                    for state in states:
                        arg_list = []
                        for item in mean_log[litmus_name_without_suffix][str(state)]:
                            print(item)
                            print(mean_log[litmus_name_without_suffix][str(state)][item][1], mean_log[litmus_name_without_suffix][str(state)][item][0])
                            arg_list.append((item,mean_log[litmus_name_without_suffix][str(state)][item][1]/mean_log[litmus_name_without_suffix][str(state)][item][0], mean_log[litmus_name_without_suffix][str(state)][item][1]/mean_log[litmus_name_without_suffix][str(state)][item][2]))
                        print(state)
                        for item in arg_list:
                            print(item)
                        log[litmus_name_without_suffix][str(state)] = arg_list
    for litmus_name_without_suffix in log:
        for state in log[litmus_name_without_suffix]:
            if not mean_flag:
                txt_dir = os.path.join(output_dir, f'{litmus_name_without_suffix}_{state}.txt')
            else:
                txt_dir = os.path.join(output_dir, f'mean_{litmus_name_without_suffix}_{state}.txt')
            with open(txt_dir, 'w') as f:
                sorted_data = sorted(log[litmus_name_without_suffix][state], key=lambda x: int(x[1]))
                for litmus_name, state_num, timecost in sorted_data:
                    f.write(litmus_name+' : '+str(state_num)+ " "+ str(timecost)+'\n')
                    f.write('\n')

def parse_perple(file):
    with open(file, 'r') as f:
        text = f.read()
    # --- 第一项：提取 heuristic statistic: XXX ---
    nums = re.findall(r'heuristic statistic:\s*(\d+)', text)
    sum_heuristic = sum(map(int, nums))

    # --- 第二项：提取 Time XXX YYY 中的最后一个数字 ---
    # 匹配规则：
    # Time <任意非换行字符> <数字>
    m = re.search(r'^\s*Time[^\n]*?([0-9]*\.?[0-9]+)', text, flags=re.MULTILINE)

    # time_value = float(m.group(1)) if m else None
    time_value = 0
    for line in text.splitlines():
        if "Time" in line:
            print("RAW:", repr(line))
            time_value = line.split(' ')[2]
            break
    return (sum_heuristic, time_value)

def statistic_state_for_perple_mean(statistic_dir, output_dir):
    log_files = []
    for dirpath, _, filenames in os.walk(statistic_dir):
        for filename in filenames:
            if filename.endswith(".log"):
                os.rename(os.path.join(dirpath, filename),
                          os.path.join(dirpath, filename.replace("100000+", "100000-")))
                log_files.append(os.path.join(dirpath, filename))

    log = {}
    mean_log = {}
    for log_file in log_files:
        litmus_name = log_file.split("/")[-1].split(".")[0]
        litmus_name_without_suffix = litmus_name.split("_")[0]
        if litmus_name_without_suffix not in mean_log:
            mean_log[litmus_name_without_suffix] = {}
        state_num, time = parse_perple(log_file)
        if litmus_name not in litmus_name_without_suffix:
            mean_log[litmus_name_without_suffix][litmus_name] = (litmus_name, state_num, time)
        else:
            mean_log[litmus_name_without_suffix][litmus_name] = (litmus_name, state_num + mean_log[litmus_name_without_suffix][litmus_name][1], time + mean_log[litmus_name_without_suffix][litmus_name][2])

    for litmus_name_without_suffix in mean_log:
        txt_dir = os.path.join(output_dir, f'mean_{litmus_name_without_suffix}.txt')

        # 收集所有记录
        items = []
        for litmus_name in mean_log[litmus_name_without_suffix]:
            state_num = mean_log[litmus_name_without_suffix][litmus_name][1]
            time = mean_log[litmus_name_without_suffix][litmus_name][2]
            items.append((litmus_name, state_num, time))

        # 按 state_num 排序（从小到大）
        items.sort(key=lambda x: x[1])

        # 写入文件
        with open(txt_dir, 'w') as f:
            for litmus_name, state_num, time in items:
                f.write(f"{litmus_name} : {state_num} {time}\n")
            f.write('\n')

if __name__ == '__main__':
    files = [f for f in os.listdir(litmus_dir) if os.path.isfile(os.path.join(litmus_dir, f))]
    chip_dir = '/home/sipeed/arg'
    # files = ['SB.litmus']
    # files = ['LB+SB_1.litmus']
    iter = 0
    # for litmus in files:
    #     litmus = os.path.join(litmus_dir, litmus)
    #     litmus_name = os.path.splitext(litmus)[0].split('/')[-1]
    #     print(litmus_name)
    #     build_userfence_cmd(os.path.join(userfence_dir, litmus_name), litmus)
    #     # build_none_cmd(os.path.join(none_dir, litmus_name), litmus)
    #     iter += 1
    #     print(iter)

    # for litmus in files:
    #     litmus = os.path.join(litmus_dir, litmus)
    #     litmus_name = os.path.splitext(litmus)[0].split('/')[-1]
    #     print(litmus_name)
    #     # run_perple_in_chip(os.path.join(perple_dir, litmus_name),chip_dir, litmus_name)
    #     run_userfence_in_chip(os.path.join(arg_dir, litmus_name), chip_dir, litmus_name)
        # run_none_in_chip(os.path.join(none_dir, litmus_name), chip_dir, litmus_name)

    for root, dirs, files in os.walk(arg_dir):
        for d in dirs:
            if d == "log":
                continue
            litmus_path = os.path.join(root, d)
            litmus_name = d.split("_")[0]
            mode = "_".join(d.split("_")[1:])
            print(litmus_name, mode, iter)
            # if os.path.exists(os.path.join(chip_dir, f"{d}/{d}-100000-{0}")):
            #     continue

            run_arg_in_chip(litmus_path,chip_dir, litmus_name, mode, 0)
            iter += 1

    # statistic_state(arg_dir, os.path.join(arg_dir, 'log'))
    # statistic_state(arg_dir, os.path.join(arg_dir, 'log'), True)

    # statistic_state_for_perple(arg_dir, os.path.join(arg_dir, 'log'))
    statistic_state_for_perple_mean(arg_dir, os.path.join(arg_dir, 'log'))
