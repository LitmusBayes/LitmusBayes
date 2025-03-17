import os

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


def make_perple_cmd(out_dir, litmus_file):
    litmus_name = litmus_file.split("/")[-1].split(".")[0]
    input_path = os.path.join(out_dir, f'{litmus_name}.c')
    filter_Thread(input_path, input_path, litmus_file, mode = 'h')
    run_cmd(f'cd {out_dir};make')

def run_ssh(input_dir, chip_dir, litmus_name, mode = "", run_time = 100):
    import paramiko

    # 配置 SSH 登录信息
    host = "192.168.156.168"  # 远程服务器地址
    port = 22  # SSH 端口
    username = "sipeed"  # SSH 用户名
    password = "sipeed"  # SSH 密码

    # 文件路径
    local_file = os.path.join(input_dir, f"run.exe")  # 本地要上传的文件
    remote_file = os.path.join(chip_dir, "run.exe")  # 上传到远程的路径
    remote_log = os.path.join(chip_dir, f"{litmus_name}_{mode}_{run_time}.log")  # 远程日志文件
    local_log = os.path.join(input_dir, f"{litmus_name}_{mode}_{run_time}.log")  # 下载回本地的路径

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


test_dir = '/home/whq/Desktop/code_list/perple_test'
# litmus_dir = os.path.join(test_dir, 'litmus_test')
litmus_dir = os.path.join(test_dir, 'idea-test')
perple_dir = os.path.join(test_dir, 'perple')
userfence_dir = os.path.join(test_dir, 'userfence')
none_dir = os.path.join(test_dir, 'none')
arg_dir = os.path.join(test_dir, 'arg')

if __name__ == '__main__':
    # files = [f for f in os.listdir(litmus_dir) if os.path.isfile(os.path.join(litmus_dir, f))]
    chip_dir = '/home/sipeed/perple'
    # files = ['SB.litmus']
    files = ['LB+SB_1.litmus']

    for litmus in files:
        if litmus in ['PPOCA_loop2.litmus']:
            continue
        litmus = os.path.join(litmus_dir, litmus)
        litmus_name = os.path.splitext(litmus)[0].split('/')[-1]
        print(litmus_name)
        # build_arg_cmd(os.path.join(arg_dir, litmus_name), litmus)
        # build_perple_cmd(os.path.join(perple_dir, litmus_name), litmus)
        build_userfence_cmd(os.path.join(userfence_dir, litmus_name), litmus)
        # build_none_cmd(os.path.join(none_dir, litmus_name), litmus)


    for litmus in files:
        litmus = os.path.join(litmus_dir, litmus)
        litmus_name = os.path.splitext(litmus)[0].split('/')[-1]
        print(litmus_name)
        # run_perple_in_chip(os.path.join(perple_dir, litmus_name),chip_dir, litmus_name)
        run_userfence_in_chip(os.path.join(userfence_dir, litmus_name), chip_dir, litmus_name)
        # run_none_in_chip(os.path.join(none_dir, litmus_name), chip_dir, litmus_name)








