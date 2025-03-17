import subprocess
import time

from src.litmusbayes.log import *
from src.litmusbayes.utils import regex_util


def run_cmd(cmd):
    return run_cmd_with_output(cmd)


def run_cmd_to_log(cmd, out_file, err_file):
    start_time = time.time()
    INFO(f"cmd to run: {cmd}")
    subprocess.call(cmd, shell=True, stdout=out_file, stderr=err_file)
    INFO(f"cmd execution time: {time.time() - start_time}")

def run_cmd_and_log(cmd):
    print(cmd)
    process = subprocess.Popen(
        cmd,
        shell=True
    )
    process.wait()


def run_cmd_with_output(cmd):
    start_time = time.time()
    INFO(f"cmd to run: {cmd}")
    p = subprocess.run(cmd,
                       shell=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    try:
        output = p.stdout.decode('utf-8')
    except UnicodeDecodeError:
        logger.warning("cmd UnicodeDecodeError")
        output = p.stdout.decode('unicode_escape')

    error = p.stderr.decode('utf-8')
    if len(error) > 0 and error.strip() != "Picked up JAVA_TOOL_OPTIONS: -Dfile.encoding=UTF8":
        # ERROR(f"output error: {error}")
        pass
    if len(output) > 0:
        INFO(f"output of this cmd: {output}")

    INFO(f"cmd execution time: {time.time() - start_time}")
    return output


def run_cmd_with_output_without_log(cmd):
    """
    only return output, with no cmd basic info print to the console
    """
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output = p.stdout.decode('utf-8')
    except UnicodeDecodeError:
        ERROR("cmd UnicodeDecodeError")
        output = p.stdout.decode('unicode_escape')

    return output


def run_cmd_without_output(cmd):
    start_time = time.time()
    INFO(f"cmd to run: {cmd}")
    return_code = subprocess.call(cmd, shell=True)
    INFO(f"cmd execution time: {time.time() - start_time}")
    return return_code


def run_cmd_on_windows(cmd):
    """
    run_cmd does not work for windows. so create a new function here.

    refs: https://www.linuxscrew.com/python-subprocesses
    """
    # output = subprocess.run(["uname", "-xyz"], capture_output=True, text=True)
    output = subprocess.run(regex_util.split_spaces(cmd), capture_output=True, text=True)
    print("STDOUT: ", output.stdout)
    print("STDERR: ", output.stderr)


def clone_repo(url, repo_dir):
    """
    Example:
            repo_dir = os.path.join(cur_dir, 'litmus-tests-riscv')
            url = "git@gitee.com:dehengyang/litmus-tests-riscv.git"
            if not os.path.exists(repo_dir):
                start_ssh_cmd = r'eval "$(ssh-agent)" && ssh-add ~/.ssh/id_rsa_deheng_wsl'
                Cmd_util.run_cmd(f"{start_ssh_cmd} && git clone {url} {repo_dir}")
    """
    cmd = f"""
    git clone {url} {repo_dir}
    """
    if is_windows():
        run_cmd_on_windows(cmd)
    else:
        run_cmd_with_output(cmd)


def is_windows():
    platform = sys.platform
    if platform == "win32":
        return True
    return False
