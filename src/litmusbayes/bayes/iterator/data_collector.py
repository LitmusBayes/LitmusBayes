import json
import logging
import os
import random
import threading
import time
from collections import defaultdict

from src.litmusbayes.bayes.LitmusPipeline import LitmusPipeline, SharedResourceManager
from src.litmusbayes.bayes.framework import LitmusRunner
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files, parse_log_by_mode_perple, parse_log_by_mode
from src.litmusbayes.utils.cmd_util import run_cmd

SEED = 2025
LOG_NAME = "collector_run"


# ================= 结果缓存类 =================
class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        # 为了避免多线程写入冲突，这里只做内存记录和追加写入，不做复杂逻辑
        # 实际的合并去重交给主控脚本的 aggregator 处理
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        key = self._make_key(obj["litmus"], obj["param"])
                        self.data[key] = obj["score"]
                    except:
                        pass
        self.f = open(path, "a")
        self.lock = threading.Lock()

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        key = self._make_key(litmus, param_vec)
        with self.lock:
            if key in self.data: return
            self.data[key] = score
            self.f.write(json.dumps({
                "litmus": litmus, "param": param_vec, "score": score
            }) + "\n")
            self.f.flush()


# ================= 任务执行类 =================
class RandomGridRunner(LitmusRunner):
    def __init__(
            self,
            shared_todo_queue,
            param_space,
            stat_log,
            perple_dict_path,
            resource_manager=None,
            mode="time",
            pipeline_host="192.168.1.105",
            pipeline_user="root",
            pipeline_pass="riscv",
            pipeline_port=22,
            litmus_path_base=None,
            litmus_dir_path=None,
            remote_work_dir=None
    ):
        super().__init__([], [], stat_log, mode)
        self.litmus_path_base = litmus_path_base
        self.litmus_dir_path = litmus_dir_path
        self.remote_work_dir = remote_work_dir

        self.ps = param_space
        self.log_dir = f"{litmus_dir_path}_{pipeline_host}"
        run_cmd(f"mkdir -p {self.log_dir}")
        unique_logger_name = f"runner_{pipeline_host}"
        self.logger = setup_logger(
            log_file=stat_log,
            level=logging.INFO,
            name=unique_logger_name,
            stdout=True  # 建议设为 False，只看文件；设为 True 则控制台也会输出
        )

        # self.logger = get_logger(LOG_NAME)
        self.resource_manager = resource_manager

        self.todo_queue = shared_todo_queue

        # 打乱顺序，避免同一个 litmus 连续跑导致板子过热或其他偏差（可选）
        # random.shuffle(self.todo_queue)

        # self.logger.info(
        #     f"Total tasks generated: {len(self.todo_queue)} (Litmus files: {len(litmus_list)} * Vectors: {num_random_vectors})")

        self.cache = ResultCache(stat_log + ".cache.jsonl")
        perple_files = get_files(perple_dict_path, "jsonl")
        self.perple_dict_list = {}
        for perple_file in perple_files:
            with open(perple_file, "r") as f:
                loaded_dict = json.load(f)
                self.perple_dict_list[perple_file.split("/")[-1][:-7]] = loaded_dict
        # 初始化流水线
        self.logger.info("Initializing Async Pipeline...")
        self.pipeline = LitmusPipeline(
            host=pipeline_host,
            port=pipeline_port,
            username=pipeline_user,
            password=pipeline_pass,
            logger = self.logger,
            resource_manager=self.resource_manager,
            remote_work_dir=remote_work_dir
        )
        self.pipeline.start(compiler_thread_count=2, downloader_thread_count=2)

    def _submit_one(self, litmus, vec, perp_dict):
        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()
        # 把 vector 挂载到 params 上，方便结果回来时找回
        params._temp_vec = vec

        litmus_file = f"{self.litmus_path_base}/{litmus}.litmus"

        self.pipeline.submit_task(
            litmus_path=litmus_file,
            params=params,
            litmus_dir_path=self.litmus_dir_path,
            log_dir_path=self.log_dir,
            run_time=100000,
            perp_dict=perp_dict
        )

    def _parse_log_to_score(self, log_path, litmus_name, has_perple, mode):
        try:
            if has_perple:
                return parse_log_by_mode_perple(log_path, mode)
            else:
                return parse_log_by_mode(log_path, mode)
        except Exception as e:
            self.logger.error(f"Parse error {log_path}: {e}")
            return -1

    def run(self):
        self.logger.info("=== Starting Robust Random Grid Execution Loop ===")

        # === 关键配置 ===
        # Pipeline 里的 keep_fresh 默认阈值是 20。
        # 我们必须让 SUBMIT_LIMIT < 20。
        # 否则 Pipeline 可能会静默丢弃任务，导致我们这边收不到结果，计数器无法归零，程序卡死。
        MAX_IN_FLIGHT = 16

        # 计数器
        active_count = 0  # 当前在 Pipeline 处理中的任务数
        finished_count = 0  # 已完成并拿到结果的任务数
        skipped_count = 0  # 命中 Cache 跳过的任务数
        # 辅助函数：填充流水线
        def try_fill_pipeline():
            nonlocal active_count
            # 只要 1. 还有待办任务 且 2. 流水线没塞满
            while len(self.todo_queue) > 0 and active_count < MAX_IN_FLIGHT:
                litmus, vec = self.todo_queue.pop(0)

                # A. 查重 (Cache Hit)
                # 如果之前跑过，就不发给 Pipeline 了，直接跳过
                if self.cache.get(litmus, vec) is not None:
                    # self.logger.info(f"[SKIP] {litmus} already cached.") # 日志太多可注释掉
                    nonlocal skipped_count
                    skipped_count += 1
                    continue

                # B. 提交任务
                self._submit_one(litmus, vec, self.perple_dict_list.get(litmus, None))
                active_count += 1

        # --- 1. 初始启动 ---
        # 先塞满 MAX_IN_FLIGHT 个任务，让 Pipeline 动起来
        self.logger.info(f"Initial filling (Target flight size: {MAX_IN_FLIGHT})...")
        try_fill_pipeline()

        if active_count == 0 and len(self.todo_queue) == 0:
            self.logger.info("Nothing to run (All cached or empty queue).")
            return

        # --- 2. 事件循环 (基于 Generator) ---
        # stream_results 会阻塞等待，直到有结果产生
        for result in self.pipeline.stream_results():

            # === A. 处理结果 ===
            active_count -= 1  # 关键：收到一个结果，飞行数减一
            finished_count += 1

            task = result['task']
            log_path = result['log_path']
            litmus_name = task.litmus_path.split("/")[-1][:-7]

            if hasattr(task.params, '_temp_vec'):
                param_vec = task.params._temp_vec
                # 算分
                score = self._parse_log_to_score(log_path, litmus_name, task.params.is_perple(), self.mode)
                self.logger.info(
                    f"[DONE] {litmus_name} | Score: {score:.4f} | In-flight: {active_count} | Rem: {len(self.todo_queue)}")

                # 存 Cache
                self.cache.add(litmus_name, param_vec, score)
            else:
                self.logger.error("Lost vector info in task result!")

            # === B. 补充新任务 (Refill) ===
            # 因为刚才腾出了一个位置 (active_count -= 1)，所以尝试补货
            try_fill_pipeline()

            # === C. 退出检查 ===
            # 如果待办队列空了，且流水线里也没任务了，说明全部做完
            if len(self.todo_queue) == 0 and active_count == 0:
                self.logger.info("Queue empty and Pipeline drained. Exiting loop.")
                break

        # --- 3. 收尾 ---
        self.logger.info(f"Run Finished. Total Processed: {finished_count}, Skipped (Cached): {skipped_count}")

        # 通知 Pipeline 停止内部线程
        self.pipeline.wait_completion()



# ================= 入口函数 =================

from queue import Queue

perple_dict_path = "../perple_json"
def main_collector(task_list, board_configs, base_config, iter):
    """
    Args:
        task_list: list of dict {'litmus': name, 'param': vec}
        board_configs: list of dict (host, user, pass...)
        base_config: dict (litmus_path_base, stat_log_dir, ...)
    """
    random.seed(SEED)
    ts = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        log_file=f"{base_config['stat_log_dir']}/collector_master.{ts}.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )

    logger.info(f"=== [Collector] Start | Tasks: {len(task_list)} | Boards: {len(board_configs)} ===")

    # 1. 构建共享任务队列
    # 使用 Queue 保证多线程安全
    task_queue = []
    for t in task_list:
        task_queue.append((t['litmus'], t['param']))

    logger.info("Task queue populated.")

    param_space = LitmusParamSpace()
    global_manager = SharedResourceManager(total_consumers=len(board_configs))

    threads = []

    for board in board_configs:
        # 每个板子单独的 stat log
        board_stat_log = f"{base_config['stat_log_dir']}/log_record_{board['host']}-{iter}.log"

        runner = RandomGridRunner(
            shared_todo_queue=[item for item in task_queue],
            param_space=param_space,
            stat_log=board_stat_log,
            perple_dict_path=perple_dict_path,
            resource_manager=global_manager,
            pipeline_host=board['host'],
            pipeline_user=board['user'],
            pipeline_pass=board['pass'],
            pipeline_port=board['port'],
            litmus_path_base=base_config['litmus_path_base'],
            litmus_dir_path=base_config['litmus_dir_path'],
            remote_work_dir=base_config['remote_work_dir']
        )

        t = threading.Thread(target=runner.run)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logger.info("All boards finished.")