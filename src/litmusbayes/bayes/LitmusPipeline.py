import os
import shutil
import time
import uuid
import threading
import queue
from collections import defaultdict

import paramiko
from dataclasses import dataclass

from filelock import FileLock

from src.litmusbayes.perple.trans_cpp import filter_Thread
from src.litmusbayes.utils.cmd_util import run_cmd
import logging

# 假设 params 和 run_cmd 等外部依赖已经定义
# from your_module import LitmusParams, run_cmd, ...

@dataclass
class LitmusTask:
    litmus_path: str
    params: object  # LitmusParams
    litmus_dir_path: str
    log_dir_path: str
    run_time: int

    # 运行时生成的属性
    unique_id: str = None
    local_exe_path: str = None
    remote_exe_path: str = None
    remote_log_path: str = None
    local_log_path: str = None
    litmus_dir: str = None
    perp_dict: dict = None


class SharedResourceManager:
    def __init__(self, total_consumers):
        """
        total_consumers: 总共有多少个板子在跑 (例如 8)
        """
        self.total = total_consumers
        self.finished_counts = defaultdict(int)  # 记录每个文件夹有多少人跑完了
        self.lock = threading.Lock()  # 线程锁，保证计数安全

    def mark_done(self, folder_path):
        """
        调用此方法表示：有一个板子已经用完这个文件夹了
        """
        should_delete = False
        with self.lock:
            self.finished_counts[folder_path] += 1
            current_count = self.finished_counts[folder_path]

            # 如果跑完的数量 >= 总板子数，说明没人用了，可以删
            if current_count >= self.total:
                should_delete = True
                # 从字典里移除，防止字典无限膨胀
                del self.finished_counts[folder_path]

        if should_delete:
            if os.path.exists(folder_path):
                print(f"[Manager] All {self.total} boards finished. Deleting: {folder_path}")
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"[Manager] Delete failed: {e}")


class LitmusPipeline:
    def __init__(self, host, port, username, password, resource_manager = None, logger=None, remote_work_dir="/tmp/litmus_work"):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.remote_work_dir = remote_work_dir
        self.logger = logger
        self.compile_queue = queue.Queue()
        self.upload_queue = queue.Queue()
        self.run_queue = queue.Queue()
        self.download_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = True
        self.resource_manager = resource_manager
        self.clean_remote_env()
        # 为了防止大量并发连接瞬间把 SSH Server 冲垮，这里加一个连接数的信号量限制
        # 比如限制同时最多 10 个 SSH 连接
        self.ssh_semaphore = threading.Semaphore(10)

    def clean_remote_env(self):
        """
        在流水线启动前，强制清空远程工作目录。
        """
        self.logger.info(f"[Pipeline] Cleaning remote directory: {self.remote_work_dir} ...")
        ssh = None
        try:
            # 建立一个临时连接用于清理
            ssh = self._get_fresh_ssh()

            # 执行清空命令：
            # 1. rm -rf 删除整个目录 (防止里面有上次残留的只读文件等)
            # 2. mkdir -p 重新创建目录
            # 注意：这会删除 remote_work_dir 下的所有内容，请确保路径配置正确！
            cmd = f"rm -rf {self.remote_work_dir} && mkdir -p {self.remote_work_dir}"

            stdin, stdout, stderr = ssh.exec_command(cmd)

            # 等待命令执行完成，确保目录真的重建好了再继续
            exit_status = stdout.channel.recv_exit_status()

            if exit_status == 0:
                self.logger.info(f"[Pipeline] Remote environment ready.")
            else:
                self.logger.info(f"[Pipeline] Warning: Clean command exited with status {exit_status}")

        except Exception as e:
            self.logger.info(f"[Pipeline] Critical Error cleaning remote env: {e}")
            # 如果连环境都清空不了，可能需要抛出异常终止程序，或者视情况而定
            raise e
        finally:
            if ssh:
                ssh.close()

    def _get_fresh_ssh(self):
        """辅助函数：创建一个全新的 SSH 连接"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # 设置 timeout 防止网络卡死
        ssh.connect(self.host, self.port, self.username, self.password, timeout=10)
        return ssh

    def stream_results(self):
        """
        这是一个生成器。
        主线程调用它时，就像在遍历一个永远读不完的列表。
        """
        while self.running:  # 只要流水线没关，就一直循环
            try:
                # 1. 尝试从结果队列里拿东西
                # timeout=1 表示：如果队列空了，我只等 1 秒。
                # 如果 1 秒后还是空的，抛出 Empty 异常，让我有机会检查 self.running 还在不在。
                result = self.result_queue.get(timeout=1)

                # 2. 拿到结果了！把结果“吐”给主线程
                # 此时，stream_results 函数会在这里“暂停”！
                # 直到主线程处理完这个 result，进入下一次循环，这里才会“恢复”继续运行。
                yield result

            except queue.Empty:
                # === [修改点 1]：查看当前各队列积压情况 ===
                # 获取各个队列的当前大小 (qsize 是近似值，但够用了)
                c_q = self.compile_queue.qsize()
                u_q = self.upload_queue.qsize()
                r_q = self.run_queue.qsize()
                d_q = self.download_queue.qsize()
                res_q = self.result_queue.qsize()

                # 打印状态（建议用 debug 或 info，这里用 self.logger.info 为了直观）
                # 只有当队列里还有东西，或者你觉得需要监控时才打日志，防止刷屏
                # 但为了调试死锁，我们先无条件打印，或者只在全部为空时打印

                # 你的需求：每次进入异常都看一下
                self.logger.info(f"[Pipeline Status] Compile:{c_q}, Upload:{u_q}, Run:{r_q}, Down:{d_q}, Result:{res_q}")

                # === [修改点 2]：如果所有队列都空了，吐出 None ===
                if c_q == 0 and u_q == 0 and r_q == 0 and d_q == 0 and res_q == 0:
                    # 只有当真的没有任何任务在跑了，才通知主线程“我饿了/空闲了”
                    # 这样主线程收到 None，就可以决定是否要补充随机任务，或者退出
                    continue

                # 如果队列不为空（比如还有任务在 Run），那就继续等，不需要 yield None 干扰主线程
                # (或者如果你希望主线程每秒都能动一下，无论是否空闲，都在这里 yield None 也可以)
                continue


    def keep_fresh(self, max_size=20):
        """
        [修正版] 源头截流：清理待编译队列。
        如果待编译的任务太多，说明后端堵死了，直接把积压在编译队列里的旧任务扔掉。
        这样既省了编译 CPU，又省了上传带宽。
        """
        # 1. 检查 compile_queue (源头)
        q_size = self.compile_queue.qsize()

        # 我们希望保留一定的 buffer，但不要太多
        # 注意：这里我们设定 compile_queue 的阈值。
        # 如果 Runner 很慢，Compile Queue 会迅速堆积。
        if q_size > max_size:
            discard_count = q_size - max_size
            self.logger.info(f"[Pipeline] System congested. Discarding {discard_count} stale tasks from COMPILE queue...")

            for _ in range(discard_count):
                try:
                    # get() 拿出来的是最老的任务（FIFO）
                    task = self.compile_queue.get_nowait()

                    # 直接丢弃，不做任何处理
                    # 也不需要删文件，因为文件根本还没生成！
                    self.logger.info(f"   -> Dropped task {task.unique_id} (Saved compile time)")

                    self.compile_queue.task_done()
                except queue.Empty:
                    break

    def _connect_ssh(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.host, self.port, self.username, self.password)
        # 预先创建远程工作目录
        ssh.exec_command(f"mkdir -p {self.remote_work_dir}")
        return ssh

    def submit_task(self, litmus_path, params, litmus_dir_path, log_dir_path, run_time=100, perp_dict=None):
        """提交任务入口"""
        task = LitmusTask(litmus_path, params, litmus_dir_path, log_dir_path, run_time)
        task.unique_id = str(uuid.uuid4())[:8]  # 生成短UUID防止文件名冲突
        task.perp_dict = perp_dict
        self.compile_queue.put(task)

    # --- Worker 1: 编译器 (本地 CPU 密集型) ---
    def worker_compiler(self):
        make_jobs = 4  # 根据你的 CPU 核心数调整

        while self.running:
            try:
                task = self.compile_queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                # self.logger.info(f"[Compiler] Building {task.litmus_path}...") # 日志太多可注释
                litmus_name = task.litmus_path.split("/")[-1][:-7]

                # 确保路径存在
                task.litmus_dir = os.path.join(task.litmus_dir_path, f"{litmus_name}_{str(task.params)}")
                exe_path = os.path.join(task.litmus_dir, "run.exe")

                # 预先赋值路径，防止下面异常导致路径为空
                task.local_exe_path = exe_path
                task.remote_exe_path = os.path.join(self.remote_work_dir, f"run_{task.unique_id}.exe")
                task.remote_log_path = os.path.join(self.remote_work_dir, f"run_{task.unique_id}.log")

                if not os.path.exists(task.litmus_dir):
                    os.makedirs(task.litmus_dir, exist_ok=True)

                lock_path = os.path.join(task.litmus_dir, "build.lock")

                # 使用 try-catch 保护业务逻辑
                compile_success = False

                with FileLock(lock_path):
                    if not os.path.exists(exe_path):
                        # 1. 生成 Litmus7 代码
                        # 建议加 try-catch 保护 run_cmd
                        gen_cmd_str = task.params.to_litmus7_format(task.litmus_path, task.litmus_dir)
                        run_cmd(gen_cmd_str)

                        # 2. 执行 Filter Thread (如果是 Perple 模式)
                        if task.params.is_perple():
                            c_file = f"{task.litmus_dir}/{litmus_name}.c"
                            # 检查源文件是否存在
                            if os.path.exists(c_file):
                                filter_Thread(c_file, c_file, task.litmus_path, task.perp_dict)
                            else:
                                raise FileNotFoundError(f"Source file missing: {c_file}")

                        # 3. 执行 Make
                        # 检查 Makefile 是否生成了，防止 make 报错
                        if not os.path.exists(os.path.join(task.litmus_dir, "Makefile")):
                            raise FileNotFoundError(f"Makefile not found in {task.litmus_dir}")

                        build_cmd = f"cd {task.litmus_dir}; make -j{make_jobs} > /dev/null 2>&1"
                        ret_code = os.system(build_cmd)

                        if ret_code != 0:
                            raise RuntimeError(f"Make failed with code {ret_code}")
                    else:
                        # Cache Hit
                        pass

                    # 再次确认 exe 是否真的存在
                    if os.path.exists(exe_path):
                        compile_success = True
                    else:
                        raise RuntimeError("Make finished but run.exe not found")

                if compile_success:
                    self.upload_queue.put(task)
                else:
                    self.logger.error(f"[Compiler] Failed to compile {task.unique_id}, dropping task.")
                    # 这里可以选择是否把失败结果放入 result_queue 通知主线程，或者直接丢弃

            except Exception as e:
                # [关键] 捕获所有逻辑错误，防止线程退出
                self.logger.error(f"[Compiler] CRITICAL ERROR on {task.unique_id}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())  # 打印详细堆栈，帮你找 bug
            finally:
                # [关键] 无论成功失败，都要通知队列该任务已结束
                self.compile_queue.task_done()

    # def worker_uploader(self):
    #     while self.running:
    #         try:
    #             task = self.upload_queue.get(timeout=2)
    #         except queue.Empty:
    #             continue
    #
    #         self.logger.info(f"[Uploader] Connecting & Uploading {task.unique_id}...")
    #
    #         # 使用 semaphore 控制并发连接数，防止 "Max Startups" 报错
    #         with self.ssh_semaphore:
    #             ssh_client = None
    #             sftp_client = None
    #             try:
    #                 # 1. 现场建立连接
    #                 ssh_client = self._get_fresh_ssh()
    #                 sftp_client = ssh_client.open_sftp()
    #
    #                 # 2. 也是先创建远程目录（虽然有点冗余，但最保险）
    #                 # 或者你可以在 Init 里只连一次创建目录，这里就不管了
    #                 # sftp_client.mkdir(self.remote_work_dir) # 可能会报错如果已存在，忽略即可
    #
    #                 # 3. 传文件 & 改权限
    #                 sftp_client.put(task.local_exe_path, task.remote_exe_path)
    #                 ssh_client.exec_command(f"chmod +x {task.remote_exe_path}")
    #
    #                 # 4. 成功入队
    #                 self.run_queue.put(task)
    #
    #             except Exception as e:
    #                 self.logger.info(f"[Uploader] Error on {task.unique_id}: {e}")
    #                 # 失败了可以选择重试或者记录错误，这里简单处理
    #             finally:
    #                 # 【重要】用完必须关，否则文件句柄泄露
    #                 if sftp_client: sftp_client.close()
    #                 if ssh_client: ssh_client.close()
    #
    #         self.upload_queue.task_done()
    def worker_uploader(self):
        # 建议设置一个单次最大批处理数量，防止一次hold住锁太久
        BATCH_SIZE = 50

        while self.running:
            # --- 1. 收集本批次要处理的任务 ---
            batch_tasks = []
            try:
                # 阻塞等待第一个任务（如果队列为空，Worker 在这里休眠）
                first_task = self.upload_queue.get(timeout=2)
                batch_tasks.append(first_task)

                # 尝试取出队列中剩余的所有任务（或者直到达到 BATCH_SIZE）
                # 注意：这里使用 get_nowait()，它不会阻塞
                while len(batch_tasks) < BATCH_SIZE:
                    try:
                        t = self.upload_queue.get_nowait()
                        batch_tasks.append(t)
                    except queue.Empty:
                        break  # 队列空了，就处理手头这些
            except queue.Empty:
                continue  # 如果超时还没等到第一个任务，重新循环

            self.logger.info(f"[Uploader] Processing batch of {len(batch_tasks)} tasks...")

            # --- 2. 建立一次连接，处理一批任务 ---
            with self.ssh_semaphore:
                ssh_client = None
                sftp_client = None
                try:
                    # === 建立连接 (只做一次) ===
                    ssh_client = self._get_fresh_ssh()
                    sftp_client = ssh_client.open_sftp()

                    # === 循环处理任务 ===
                    for task in batch_tasks:
                        try:
                            self.logger.info(f"  -> Uploading {task.unique_id}...")

                            # 传输文件
                            sftp_client.put(task.local_exe_path, task.remote_exe_path)
                            # 修改权限 (复用 ssh 连接执行命令)
                            ssh_client.exec_command(f"chmod +x {task.remote_exe_path}")

                            # 成功入队到 Runner
                            self.run_queue.put(task)

                        except Exception as e_inner:
                            # 单个任务失败，不要打断整个循环，记录错误即可
                            self.logger.info(f"[Uploader] Failed task {task.unique_id}: {e_inner}")
                            # 这里可以做一些重试逻辑，或者把 task 丢回 upload_queue
                        finally:
                            # 标记该任务在 upload_queue 中已完成
                            self.upload_queue.task_done()

                except Exception as e_outer:
                    # 如果是连接层面的大崩溃（比如 SSH 连不上），这批任务都需要处理
                    self.logger.info(f"[Uploader] Connection Error: {e_outer}")
                    # 策略：因为还没上传成功，理论上应该把这批 batch_tasks 重新放回队列
                    for task in batch_tasks:
                        # 注意：这里需要防止死循环，实际工程中可能需要计数器
                        self.upload_queue.put(task)
                        self.upload_queue.task_done()  # 因为上面get出来了，这里要抵消

                finally:
                    # === 关闭连接 (只做一次) ===
                    if sftp_client: sftp_client.close()
                    if ssh_client: ssh_client.close()
        # --- Worker 3: 执行器 (短链接模式) ---
    # def worker_runner(self):
    #     while self.running:
    #         try:
    #             task = self.run_queue.get(timeout=2)
    #         except queue.Empty:
    #             continue
    #
    #         self.logger.info(f"[Runner] >>> Running {task.unique_id}...")
    #
    #         with self.ssh_semaphore:
    #             ssh_client = None
    #             try:
    #                 # 1. 现场连接
    #                 ssh_client = self._get_fresh_ssh()
    #
    #                 # 2. 执行命令
    #                 cmd = f"{task.remote_exe_path} -s {task.run_time} > {task.remote_log_path} 2>&1"
    #                 # exec_command 默认是不阻塞的，但我们需要等待结果
    #                 stdin, stdout, stderr = ssh_client.exec_command(cmd)
    #
    #                 # 3. 阻塞等待结束 (这是耗时步骤，但连接必须保持着)
    #                 exit_status = stdout.channel.recv_exit_status()
    #
    #                 if exit_status == 0:
    #                     self.download_queue.put(task)
    #                 else:
    #                     self.download_queue.put(task)
    #                     self.logger.info(f"[Runner] Failed {task.unique_id} status {exit_status}")
    #
    #             except Exception as e:
    #                 self.logger.info(f"[Runner] Error: {e}")
    #             finally:
    #                 if ssh_client: ssh_client.close()
    #
    #         self.run_queue.task_done()
    #
    def worker_runner(self):
        BATCH_SIZE = 50  # 同样限制一批最大处理数量

        while self.running:
            # --- 1. 收集本批次任务 ---
            batch_tasks = []
            try:
                # 阻塞等待第一个
                first_task = self.run_queue.get(timeout=2)
                batch_tasks.append(first_task)

                # 非阻塞捞取剩余任务
                while len(batch_tasks) < BATCH_SIZE:
                    try:
                        t = self.run_queue.get_nowait()
                        batch_tasks.append(t)
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            self.logger.info(f"[Runner] Processing batch of {len(batch_tasks)} tasks...")

            # --- 2. 建立连接并批量执行 ---
            with self.ssh_semaphore:
                ssh_client = None
                try:
                    # 只建立一次连接
                    ssh_client = self._get_fresh_ssh()

                    tasks_to_requeue = []  # 用于存储因连接中断而未执行的任务

                    for task in batch_tasks:
                        try:
                            self.logger.info(f"  -> Running {task.unique_id}...")

                            cmd = f"timeout 15s {task.remote_exe_path} -s {task.run_time} > {task.remote_log_path} 2>&1"
                            # cmd = f"{task.remote_exe_path} -s {task.run_time} > {task.remote_log_path} 2>&1"

                            # 复用 ssh_client 执行命令
                            # 注意：这里会开启一个新的 Channel，但复用同一个 TCP 连接
                            stdin, stdout, stderr = ssh_client.exec_command(cmd)

                            # 阻塞等待该任务结束
                            exit_status = stdout.channel.recv_exit_status()

                            if exit_status != 0:
                                self.logger.info(f"[Runner] Task {task.unique_id} finished with status {exit_status}")

                            # 无论成功失败，都算运行完了，交给下载器去拉日志
                            self.download_queue.put(task)

                            # 标记当前任务完成
                            self.run_queue.task_done()

                        except Exception as e_inner:
                            # 如果是 SSH 连接断开了，后面的任务就没法跑了
                            if isinstance(e_inner, (paramiko.SSHException, OSError)):
                                self.logger.info(f"[Runner] Connection lost at {task.unique_id}: {e_inner}")
                                # 标记当前这个失败了，需要重试
                                tasks_to_requeue.append(task)
                                # 剩下的还没跑的任务也都要重试
                                remaining_index = batch_tasks.index(task) + 1
                                tasks_to_requeue.extend(batch_tasks[remaining_index:])
                                break  # 跳出循环
                            else:
                                # 如果只是普通的逻辑错误，记录一下，不要卡住后面的
                                self.logger.info(f"[Runner] Logic Error on {task.unique_id}: {e_inner}")
                                self.run_queue.task_done()

                    # 如果有因为断连需要重试的任务，重新放回队列
                    for t in tasks_to_requeue:
                        self.run_queue.put(t)
                        # 注意：get出来的时候计数器加了，这里重放进去算是新任务，
                        # 但之前的 get 对应的 task_done 还是要调用的，为了防止死锁，
                        # 简单做法是：对于未完成的任务，我们也调用 task_done() 抵消掉之前的 get，
                        # 然后 put 进去当作全新的任务。
                        self.run_queue.task_done()

                except Exception as e_outer:
                    self.logger.info(f"[Runner] Critical Batch Error: {e_outer}")
                    # 极端情况：连 SSH 都连不上，所有任务回滚
                    for t in batch_tasks:
                        self.run_queue.put(t)
                        self.run_queue.task_done()
                finally:
                    if ssh_client: ssh_client.close()

    def worker_downloader(self):
        BATCH_SIZE = 50

        while self.running:
            # --- 1. 收集任务 ---
            batch_tasks = []
            try:
                first_task = self.download_queue.get(timeout=2)
                batch_tasks.append(first_task)
                while len(batch_tasks) < BATCH_SIZE:
                    try:
                        t = self.download_queue.get_nowait()
                        batch_tasks.append(t)
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            self.logger.info(f"[Downloader] Processing batch of {len(batch_tasks)} tasks...")

            # --- 2. 批量下载 ---
            with self.ssh_semaphore:
                ssh_client = None
                sftp_client = None
                try:
                    # 建立连接 (一次)
                    ssh_client = self._get_fresh_ssh()
                    sftp_client = ssh_client.open_sftp()

                    tasks_to_requeue = []

                    for task in batch_tasks:
                        try:
                            # 计算本地路径
                            litmus_name = task.litmus_path.split("/")[-1][:-7]  # 假设你的命名逻辑
                            task.local_log_path = os.path.join(
                                task.log_dir_path,
                                f"{litmus_name}_{str(task.params)}_{task.run_time}-{task.unique_id}.log"
                            )

                            # 执行下载
                            # self.logger.info(f"  -> Downloading {task.unique_id}...")
                            sftp_client.get(task.remote_log_path, task.local_log_path)

                            # 清理远程文件 (这也是 IO 操作，放这里顺手做了)
                            try:
                                sftp_client.remove(task.remote_log_path)
                                sftp_client.remove(task.remote_exe_path)
                            except IOError:
                                pass  # 文件不存在就算了

                            # 本地清理和输出结果
                            if self.resource_manager:
                                self.resource_manager.mark_done(task.litmus_dir)
                            else:
                                # 如果没有 manager (单机模式)，还是保持原来的直接删除
                                # 建议用 shutil 而不是 os.system，更安全
                                import shutil
                                if os.path.exists(task.litmus_dir):
                                    shutil.rmtree(task.litmus_dir)
                            # os.system(f"rm -rf {task.litmus_dir}")  # 慎用 system，建议用 shutil.rmtree

                            # 放入结果队列
                            self.result_queue.put({'task': task, 'log_path': task.local_log_path})

                            self.download_queue.task_done()

                        except Exception as e_inner:
                            # 同样处理连接中断的情况
                            if isinstance(e_inner, (paramiko.SSHException, OSError)):
                                self.logger.info(f"[Downloader] Connection lost at {task.unique_id}: {e_inner}")
                                tasks_to_requeue.append(task)
                                remaining_index = batch_tasks.index(task) + 1
                                tasks_to_requeue.extend(batch_tasks[remaining_index:])
                                break
                            else:
                                self.logger.info(f"[Downloader] Error on {task.unique_id}: {e_inner}")
                                self.download_queue.task_done()

                    # 重试逻辑
                    for t in tasks_to_requeue:
                        self.download_queue.put(t)
                        self.download_queue.task_done()

                except Exception as e_outer:
                    self.logger.info(f"[Downloader] Critical Batch Error: {e_outer}")
                    for t in batch_tasks:
                        self.download_queue.put(t)
                        self.download_queue.task_done()
                finally:
                    if sftp_client: sftp_client.close()
                    if ssh_client: ssh_client.close()

    def start(self, compiler_thread_count=2, downloader_thread_count=2):  # <--- 新增参数
        threads = []

        self.logger.info(f"Starting {compiler_thread_count} compiler threads...")
        for _ in range(compiler_thread_count):
            t = threading.Thread(target=self.worker_compiler, daemon=True)
            t.start()
            threads.append(t)

        # Uploader (可以开 2 个)
        for _ in range(2):
            t = threading.Thread(target=self.worker_uploader, daemon=True)
            t.start()
            threads.append(t)

        # Runner (必须 1 个)
        t = threading.Thread(target=self.worker_runner, daemon=True)
        t.start()
        threads.append(t)

        # Downloader (动态配置)
        self.logger.info(f"Starting {downloader_thread_count} downloader threads...")
        for _ in range(downloader_thread_count):
            t = threading.Thread(target=self.worker_downloader, daemon=True)
            t.start()
            threads.append(t)

        return threads

    def wait_completion(self):
        # 等待所有队列清空
        self.compile_queue.join()
        self.upload_queue.join()
        self.run_queue.join()
        self.download_queue.join()
        self.running = False
        self.logger.info("All tasks completed.")


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化 Pipeline
    pipeline = LitmusPipeline(
        host="192.168.1.x", port=22,
        username="user", password="pwd",
        remote_work_dir="/home/user/litmus_ci"
    )
    pipeline.start()

    # 2. 疯狂塞入任务
    # 假设你有一个任务列表 list_of_litmus_files
    # for l_path in list_of_litmus_files:
    #     pipeline.submit_task(l_path, some_params, local_dir, log_dir)

    # 3. 等待结束
    pipeline.wait_completion()