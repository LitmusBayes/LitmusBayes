import paramiko
from scp import SCPClient
import os
import sys
# ================= 配置区域 =================
# 服务器配置
# REMOTE_USER = "sipeed"  # 远程机用户名
# REMOTE_HOST = "10.42.0.131"  # 远程机 IP 或域名
# SSH_PORT = "22"  # SSH 端口
# REMOTE_PASSWORD = "sipeed"  # 在这里写死你的密码

REMOTE_USER = "root"  # 远程机用户名
REMOTE_HOST = "10.42.0.58"  # 远程机 IP 或域名
SSH_PORT = "22"  # SSH 端口
REMOTE_PASSWORD = "bianbu"  # 在这里写死你的密码

LOCAL_SOURCE_DIR = "./repos"
# REMOTE_TARGET_DIR = "/home/sipeed/perple"
# REMOTE_SCRIPT_DIR = "/home/sipeed/perple"
REMOTE_TARGET_DIR = "/root/perple"
REMOTE_SCRIPT_DIR = "/root/perple"
LOCAL_FINAL_LOG_DIR = "./Banana"

RUN_TIMES = 8
REMOTE_DEST_DIR_PARAM = "./collected_logs"


# ============================================

def create_ssh_client():
    """创建并返回一个已认证的 SSH 客户端"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 自动接受未知的主机密钥
    try:
        print(f"正在连接到 {REMOTE_HOST} 并验证密码...")
        ssh.connect(hostname=REMOTE_HOST, port=SSH_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
        print("连接与验证成功！")
        return ssh
    except paramiko.AuthenticationException:
        print("密码验证失败，请检查密码是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"连接失败: {e}")
        sys.exit(1)


def execute_command(ssh, command):
    """在远程机上执行命令并打印输出"""
    print(f"[远程执行] {command}")
    stdin, stdout, stderr = ssh.exec_command(command)

    # 实时读取输出
    for line in stdout:
        print(line.strip('\n'))

    err = stderr.read().decode().strip()
    if err:
        print(f"[错误输出]\n{err}")

    # 获取命令执行的返回码
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"命令执行失败，退出码: {exit_status}")
        sys.exit(exit_status)


def main():
    os.makedirs(LOCAL_FINAL_LOG_DIR, exist_ok=True)

    # 1. 初始化 SSH 连接
    ssh = create_ssh_client()

    try:
        # 2. 准备远程目录
        execute_command(ssh, f"mkdir -p {REMOTE_TARGET_DIR}")

        # 3. SCP 上传
        print("\n>>> 开始上传文件...")
        with SCPClient(ssh.get_transport()) as scp:
            # recursive=True 支持直接上传整个文件夹下的内容
            # 注意：此处直接传文件夹，如果本地路径是 ./dir，它会在远程创建 dir 文件夹。
            # 为了把内容铺在 REMOTE_TARGET_DIR 下，可以直接遍历本地目录上传
            for item in os.listdir(LOCAL_SOURCE_DIR):
                local_path = os.path.join(LOCAL_SOURCE_DIR, item)
                scp.put(local_path, remote_path=REMOTE_TARGET_DIR, recursive=True)
        print("上传完成！")

        # 4. 远程执行任务
        print(f"\n>>> 循环执行远程脚本 {RUN_TIMES} 次...")
        # 注意：这里改成了 _$i，利用远程 bash 的变量来实现每次目录不同
        run_cmd = (
            f"cd {REMOTE_SCRIPT_DIR} && "
            f"for i in $(seq 1 {RUN_TIMES}); do "
            f"echo '--- 运行批次: '$i' ---'; "
            f"python3 run.py && "
            f"python3 collect_log.py --DEST_DIR {REMOTE_DEST_DIR_PARAM}_$i; "
            f"done"
        )
        execute_command(ssh, run_cmd)

        # 5. SCP 下载
        print("\n>>> 开始收集日志到本地...")
        with SCPClient(ssh.get_transport()) as scp:
            # 循环下载每一次生成的独立文件夹
            for i in range(1, RUN_TIMES + 1):
                # 提取干净的文件夹名，例如 "collected_logs_1"
                base_name = f"{REMOTE_DEST_DIR_PARAM.lstrip('./')}_{i}"

                # 拼接出远程路径: /home/user/remote_work/collected_logs_1
                remote_log_dir = f"{REMOTE_SCRIPT_DIR}/{base_name}"

                # 拼接出你要求的本地目标路径: ./local_final_logs/collected_logs_1
                local_log_dir = os.path.join(LOCAL_FINAL_LOG_DIR, base_name)

                print(f"正在下载: {remote_log_dir}  --->  {local_log_dir}")
                try:
                    # 注意：传给 scp.get 的 local_path 只需要写父目录 LOCAL_FINAL_LOG_DIR 即可。
                    # 它会自动把 remote_log_dir 这个文件夹放进去，完美形成你想要的 local_log_dir 路径结构。
                    # 如果这里把 local_path 写成了 local_log_dir，反而会导致本地出现 collected_logs_1/collected_logs_1 的双重嵌套。
                    scp.get(remote_log_dir, local_path=LOCAL_FINAL_LOG_DIR, recursive=True)
                except Exception as e:
                    print(f"下载异常 {remote_log_dir}: {e}")

        print(f"日志收集完成！所有批次的文件夹已单独保存至: {LOCAL_FINAL_LOG_DIR}")
    finally:
        ssh.close()
        print("SSH 连接已关闭。")


if __name__ == "__main__":
    main()