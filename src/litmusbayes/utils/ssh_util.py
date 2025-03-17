import paramiko
from scp import SCPClient
from concurrent.futures import ThreadPoolExecutor
import sys
import os


def create_ssh_client(hostname, port, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, port, username, password)
        print(f"Connected to {hostname}")
    except Exception as e:
        print(f"Failed to connect to {hostname}: {e}")
        return None

    return ssh


def upload_file(ssh_client, local_path, remote_path):
    if ssh_client is None:
        print("SSH client is not available. Cannot upload file.")
        return

    transport = ssh_client.get_transport()
    if transport is None:
        print("Transport is None. SSH connection may not be active.")
        return

    with SCPClient(transport) as scp:
        scp.put(local_path, remote_path)
        print(f"File {local_path} uploaded to {remote_path}")


def download_file(ssh_client, remote_path, local_path):
    if ssh_client is None:
        print("SSH client is not available. Cannot download file.")
        return

    transport = ssh_client.get_transport()
    if transport is None:
        print("Transport is None. SSH connection may not be active.")
        return

    with SCPClient(transport) as scp:
        scp.get(remote_path, local_path)
        print(f"File {remote_path} downloaded to {local_path}")


def ssh_task(hostname, port, username, password, commands = None):
    error_list = []
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port, username, password)
        print(f"Connected to {hostname}")
        # with create_ssh_client(hostname, port, username, password) as ssh:
        for command in commands:
            if command['type'] == 'file_in':
                local_path, remote_path = command['field'].split(',')
                upload_file(ssh, local_path, remote_path)

            elif command['type'] == 'file_out':
                local_path, remote_path = command['field'].split(',')
                download_file(ssh, remote_path, local_path)
            else:
                print('exec', command['field'])
                stdin, stdout, stderr = ssh.exec_command(command['field'])
                output = stdout.read().decode()
                error = stderr.read().decode()

                print(f"Task on {hostname} output:\n{output}")
                if error:
                    error_list.append(error)
                    print(f"Task on {hostname} error:\n{error}")

        ssh.close()
        return True, error_list
    except Exception as e:
        print(f"Failed to execute task on {hostname}: {e}")
        return False, error_list