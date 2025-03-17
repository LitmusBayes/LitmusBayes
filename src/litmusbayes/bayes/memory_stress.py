import os
import re
import shutil
import subprocess

from src.slide import config
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy
from src.slide.utils.cmd_util import run_cmd

filter_litmus_list = [
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceptional.txt'),
]
def run_ssh(input_dir, output_dir, chip_dir, litmus_name, cmd, mode = "", time = 0, run_time = 100):
    import paramiko

    # 配置 SSH 登录信息
    host = "192.168.201.168"  # 远程服务器地址
    port = 22  # SSH 端口
    username = "sipeed"  # SSH 用户名
    password = "sipeed"  # SSH 密码

    # 文件路径
    input_file = os.path.join(input_dir, f"run.exe")
    remote_file = os.path.join(chip_dir, f'run.exe')  # 上传到远程的路径
    # remote_file = os.path.join(chip_dir, f'{litmus_name}.litmus')  # 上传到远程的路径
    remote_log = os.path.join(chip_dir, f"{litmus_name}_{mode}_{run_time}.log")  # 远程日志文件
    local_log = os.path.join(output_dir, f"{litmus_name}_{mode}_{run_time}-{time}.log")  # 下载回本地的路径

    if not os.path.exists(input_file):
        print("not" ,input_file)
        return
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
    print(f"Uploading {input_file} -> {chip_dir}")
    ssh.exec_command(f"cd {chip_dir}; rm -rf *")
    sftp.put(input_file, remote_file)


    # # 修改权限，确保可执行
    # print("Running remote command litmus7...")
    # stdin, stdout, stderr = ssh.exec_command(cmd)
    # exit_status = stdout.channel.recv_exit_status()
    # if exit_status != 0:
    #     print(f"Make failed with exit status {exit_status}")
    #     print(stderr.read().decode())
    #     return
    # print("Running remote command make...")
    # stdin, stdout, stderr = ssh.exec_command(f"cd {chip_dir}; make")
    # exit_status = stdout.channel.recv_exit_status()
    # if exit_status != 0:
    #     print(f"Make failed with exit status {exit_status}")
    #     print(stderr.read().decode())
    #     return
    # 运行命令并输出到日志文件
    print("Running remote command run...")
    ssh.exec_command(f"chmod +x {remote_file}")

    print(f"cd {chip_dir}; ./run.exe -s {run_time} > {remote_log} 2>&1")
    stdin, stdout, stderr = ssh.exec_command(f"cd {chip_dir}; ./run.exe -s {run_time} > {remote_log} 2>&1")

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

def filter_allowed_litmus(litmus_files):
    for litmus_file in litmus_files:
        cmd = f"eval $(opam env); herd7 -model riscv.cat {litmus_file}"
        print(litmus_file)
        # 运行命令并获取输出
        try:
            litmus_result = subprocess.check_output(
                cmd,
                shell=True,
                text=True,
                executable="/bin/bash"  # 重要：使用 bash 解析 eval
            )
        except subprocess.CalledProcessError as e:
            # herd7 运行失败也算结果
            litmus_result = e.output

        # 判断 Positive: 0 是否存在
        if "Positive: 0" not in litmus_result:
            print(f"[+] Found non-zero Positive in {litmus_file}, copying...")
            # 复制文件
            shutil.copy(litmus_file, "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_naive")


def get_result(in_path,out_path):
    litmus_files = [os.path.join(in_path, f) for f in os.listdir(in_path)
             if os.path.isfile(os.path.join(in_path, f))]
    no_list = []
    result_list = []
    for litmus_file in litmus_files:
        litmus_name = litmus_file.split("/")[-1][:-7]
        log_path = os.path.join(out_path, f"{litmus_name}/{litmus_name}__100000-0.log")
        if not os.path.exists(log_path):
            no_list.append(litmus_name)
            continue
        with open(log_path, "r") as fl:
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

            result_list.append((litmus_name, number, number/time))
    result_list.sort(key=lambda x: x[2])
    with open(os.path.join(out_path,"result.log"), "w") as fl:
        fl.write("not compile litmus: \n")
        fl.write("\n".join(no_list))
        fl.write("\n")
        fl.write("\n")
        fl.write("litmus result: \n")
        for litmus_name, number, times in result_list:
            fl.write(f"{litmus_name}: {number}, {times}\n")

C910_not_pass_litmus_dir = [
'R+popar+po+NEW',
'amoswap.w.aq.rl',
'2+2W+po+poarp+NEW',
'MP+poarp+po+NEW',
'S+po+poarp+NEW',
'2+2W+po+popar+NEW',
'MP+poarar+po+NEW',
'LB+po+popar+NEW',
'lr.w.aq.rl',
'AMO-FENCE',
'Luc02',
'Luc02+BIS',
'LB+po+poarp+NEW',
'SB+po+poarar+NEW',
'S+popar+po+NEW',
'S+poarar+po+NEW',
'MP+popar+po+NEW',
'R+po+poarp+NEW',
'2+2W+po+poarar+NEW',
'S+po+poarar+NEW',
'R+poarp+po+NEW',
'MP+po+popar+NEW',
'MP+po+poarp+NEW',
'SB+po+poarp+NEW',
'MP+po+poarar+NEW',
'S+poarp+po+NEW',
'LB+po+poarar+NEW',
'SB+po+popar+NEW',
'R+po+poarar+NEW',
'R+po+popar+NEW',
'S+po+popar+NEW',
]

out_path = "/home/whq/Desktop/code_list/perple_test/stress"
in_path = "/home/whq/Desktop/code_list/perple_test/stress_litmus"
chip_dir = "/home/sipeed/all"
if __name__ == "__main__":
    # litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
    #     'file_list': filter_litmus_list})
    litmus_files = [os.path.join(in_path, f) for f in os.listdir(in_path)
             if os.path.isfile(os.path.join(in_path, f))]
    # for litmus_file in litmus_files:
    #     litmus_name = litmus_file.split("/")[-1][:-7]
    #     if litmus_name in C910_not_pass_litmus_dir:
    #         continue
    #     shutil.copy(litmus_file, os.path.join(out_path, f'{litmus_name}.litmus'))

    # get_result(in_path, out_path)
    #
    counter = 0
    for litmus_file in litmus_files:
        litmus_name = litmus_file.split('/')[-1][:-7]
        print(litmus_name)
        # cmd = f"eval $(opam env); litmus7 -carch RISCV -limit true -mem direct -affinity incr1 -force_affinity true -barrier userfence -stride 1 -size_of_test 100 -number_of_run 10 -driver C -ccopts -O2 -linkopt -static -smtmode seq -smt 2  -avail 4 "
        # cmd += f" {chip_dir}/{litmus_name}.litmus"
        # cmd += f" -o {chip_dir}"
        # print(cmd)
        cmd = f"mkdir -p {out_path}/{litmus_name};eval $(opam env); litmus7 -carch RISCV -limit true -mem direct -affinity incr1 -force_affinity true -barrier userfence -stride 1 -size_of_test 100 -number_of_run 10 -driver C -ccopts -O2 -gcc riscv64-unknown-linux-gnu-gcc -linkopt -static -smtmode seq -smt 2  -avail 4 "
        cmd += f" {in_path}/{litmus_name}.litmus"
        cmd += f" -o {out_path}/{litmus_name}"
        print(cmd)
        # shutil.copy(litmus_file, os.path.join(out_path, f'{litmus_name}.litmus'))
        # run_cmd(cmd)
        # run_cmd(f"cd {out_path}/{litmus_name}; make")
        # run_ssh(litmus_file, os.path.join(out_path, litmus_name), chip_dir, litmus_name, cmd, run_time=100000)
        run_ssh(f"{out_path}/{litmus_name}", os.path.join(out_path, litmus_name), chip_dir, litmus_name, cmd, run_time=100000)
        print(f"end {counter}")
        counter += 1
    print(len(litmus_files))
    # get_result(in_path, out_path)
