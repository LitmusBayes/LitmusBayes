import os
import re
import time

import paramiko
import  pandas as pd
from collections import defaultdict


from src.litmusbayes.bayes.all_litmus_test import make_perple_cmd
from src.litmusbayes.bayes.litmus_params import LitmusParams, to_dict_by_vector
from src.litmusbayes.comp.parse_result import parse_chip_log
from src.litmusbayes.utils.cmd_util import run_cmd, run_cmd_and_log

# 配置 SSH 登录信息
# host = "192.168.226.168"  # 远程服务器地址
# host = "10.42.0.131"
# port = 22  # SSH 端口
# username = "sipeed"  # SSH 用户名
# password = "sipeed"  # SSH 密码
# remote_path = "/home/sipeed/test"


# 配置 SSH 登录信息
host = "10.42.0.58"
port = 22  # SSH 端口
username = "root"  # SSH 用户名
password = "bainbu"  # SSH 密码
remote_path = "/root/perple"


def run_litmus_by_mode(litmus_path, params: LitmusParams, litmus_dir_path, log_dir_path, run_time = 100, mode = "exe", mach = "ssh"): #mode:exe/litmus7
    litmus_name = litmus_path.split("/")[-1][:-7]
    litmus_dir = os.path.join(litmus_dir_path, f"{litmus_name}_{str(params)}")
    remote_log = os.path.join(remote_path, f"run.log")
    local_log = os.path.join(log_dir_path, f"{litmus_name}_{str(params)}_{run_time}-{time.time()}.log")  # 下载回本地的路径
    remote_exe_path = os.path.join(remote_path, f"run.exe")

    if mach == "qemu":
        exe_path = os.path.join(litmus_dir, f"run.exe")

        if not os.path.exists(exe_path):
            run_cmd(params.to_litmus7_format(litmus_path, litmus_dir))
            print(params.to_litmus7_format(litmus_path, litmus_dir))
            if params.is_perple():
                make_perple_cmd(litmus_dir, litmus_path)
            run_cmd(f"cd {litmus_dir};make")
            print(f"cd {litmus_dir};make")
        print(f"qemu-riscv64 {exe_path} -s {run_time} > {local_log}")
        run_cmd_and_log(f"qemu-riscv64 {exe_path} -s {run_time} > {local_log}")
        # run_cmd(f"qemu-riscv64 .{exe_path} -s {run_time} > {local_log}")
        return local_log

    # ssh = paramiko.SSHClient()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #
    # ssh.connect(host, port, username, password)
    # sftp = ssh.open_sftp()

    if mode == "litmus7":
        assert False, "litmus7 not supported"  # todo
    elif mode == "exe":
        pass
        exe_path = os.path.join(litmus_dir, f"run.exe")

        if not os.path.exists(exe_path):
            run_cmd(params.to_litmus7_format(litmus_path, litmus_dir))
            print(params.to_litmus7_format(litmus_path, litmus_dir))
            if params.is_perple():
                make_perple_cmd(litmus_dir, litmus_path)
            run_cmd(f"cd {litmus_dir};make")
            return -1
            print(f"cd {litmus_dir};make")
        # exe_path = os.path.join(litmus_dir, f"run.exe")

        print(f"Uploading {exe_path} -> {remote_exe_path}")

        sftp.put(exe_path, remote_exe_path)

        ssh.exec_command(f"chmod +x {remote_exe_path}")

    stdin, stdout, stderr = ssh.exec_command(f"{remote_exe_path} -s {run_time} > {remote_log} 2>&1")

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

    return local_log

def parse_log_by_mode(log_path, mode = "time"): # time/frequency
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", errors="ignore") as fl:
        result_txt = fl.readlines()
        number = -1
        time = -1
        for line in result_txt:
            match = re.search(r"Positive:\s*(\d+)", line)
            if match:
                number = int(match.group(1))
                # print(number)
            if "Time" in line:
                time = float(line.split(' ')[-1].strip())
    if time == -1 or number == -1:
        return -1
    if mode == "time":
        print(f"number={number}, time={time}")
        return number/time
    elif mode == "frequency":
        return number

    return -1

def parse_log_by_mode_perple(log_path, mode = "time"): # time/frequency
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r") as f:
        text = f.read()

        # --- 第一项：提取 heuristic statistic: XXX ---
    nums = re.findall(r'heuristic statistic:\s*(\d+)', text)
    sum_heuristic = sum(map(int, nums))

    time_value = 0
    for line in text.splitlines():
        if "Time" in line:
            print("RAW:", repr(line))
            time_value = float(line.split(' ')[2])
            break

    if time_value == 0:
        return -1
    if mode == "time":
        return sum_heuristic/time_value
    elif mode == "frequency":
        return sum_heuristic

    return -1

def get_files(directory, suffix = ".litmus"):
    return [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(suffix) and os.path.isfile(os.path.join(directory, f))
    ]



def sort_log_dict(log_dict):
    """对每个litmus_name对应的列表按第二项num排序"""
    sorted_dict = {}
    for litmus_name, data_list in log_dict.items():
        # 按元组的第二个元素num排序
        sorted_list = sorted(data_list, key=lambda x: x[1])
        sorted_dict[litmus_name] = sorted_list
    return sorted_dict



def create_excel_from_log_dict(log_dict, output_filename='output.xlsx'):
    """
    将log_dict转换为Excel文件

    Args:
        log_dict: 原始数据字典
        output_filename: 输出的Excel文件名
    """
    # 1. 先对数据进行排序
    sorted_data = sort_log_dict(log_dict)

    # 2. 准备数据用于创建DataFrame
    all_rows = []

    for litmus_name, data_list in sorted_data.items():
        for attr_dict, num in data_list:
            # 创建一行数据
            row = attr_dict.copy()
            row['litmus_name'] = litmus_name  # 添加litmus_name列
            row['num'] = num  # 添加num列
            all_rows.append(row)
    print(all_rows)
    # 3. 创建DataFrame
    df = pd.DataFrame(all_rows)

    # 4. 定义列的顺序（根据需求调整）
    # 基础列（litmus_name和num）
    base_columns = ['litmus_name', 'num']
    # attr_dict的键（按你提供的顺序）
    attr_columns = [
        'mem', 'barrier', 'alloc', 'detached', 'thread',
        'launch', 'affinity', 'stride', 'contiguous',
        'noalign', 'perple'
    ]

    # 确保所有列都存在
    all_columns = base_columns + attr_columns
    existing_columns = [col for col in all_columns if col in df.columns]

    # 重新排列列顺序
    df = df[existing_columns]

    # 5. 写入Excel
    df.to_csv(output_filename, index=False)

    print(f"Excel文件已生成: {output_filename}")
    print(f"总行数: {len(df)}")
    print(f"列数: {len(df.columns)}")

    return df




def read_log_to_summary(log_dir_path, log_path, stat_mode = "frequency"):
    log_files = get_files(log_dir_path, suffix=".log")
    log_files.sort()
    log_dict = {}
    for log_file in log_files:
        print(log_file)
        litmus_file_name = log_file.split("/")[-1]
        litmus_name = litmus_file_name.split("_")[0]
        mode = litmus_file_name.split("_")[1:-1]
        mode = [int(item) for item in mode]
        print(litmus_name, mode)
        attr_dict = to_dict_by_vector(mode)
        print(attr_dict)
        num = 0
        if attr_dict["perple"]:
            num = parse_log_by_mode_perple(log_file, stat_mode)
        else :
            num = parse_log_by_mode(log_file, stat_mode)
        print(num)
        if litmus_name not in log_dict:
            log_dict[litmus_name] = []

        log_dict[litmus_name].append((attr_dict, num))

    print(log_dict)
    create_excel_from_log_dict(log_dict, log_path)


if __name__ == "__main__":
    dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
    log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat.csv"
    read_log_to_summary(dir_path, log_path)