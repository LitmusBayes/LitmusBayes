import json
import logging
import os
import random
import threading
import time
from collections import defaultdict

# 去掉了 sklearn, scipy, torch 等不需要的库
from src.litmusbayes.bayes.LitmusPipeline import LitmusPipeline, SharedResourceManager
from src.litmusbayes.bayes.framework import LitmusRunner
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files, parse_log_by_mode_perple, parse_log_by_mode
from src.litmusbayes.utils.cmd_util import run_cmd

SEED = 2025
LOG_NAME = "random_grid"


# ================= Cache 类 (保持不变) =================
class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
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

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        key = self._make_key(litmus, param_vec)
        if key in self.data: return
        self.data[key] = score
        self.f.write(json.dumps({
            "litmus": litmus, "param": param_vec, "score": score
        }) + "\n")
        self.f.flush()


# ================= 核心运行类 (修改为随机 Grid 模式) =================
class RandomGridRunner(LitmusRunner):
    def __init__(
            self,
            litmus_list,
            param_space,
            stat_log,
            resource_manager=None,
            fixed_vectors = None,
            mode="time",
            num_random_vectors=50,  # 这里指定生成的随机向量数量
            pipeline_host="192.168.1.105",
            pipeline_user="root",
            pipeline_pass="riscv",
            pipeline_port=22
    ):
        super().__init__(litmus_list, [], stat_log, mode)
        self.ps = param_space
        self.log_dir = f"{dir_path}_{pipeline_host}"
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
        # 1. 预先生成固定的随机向量池
        self.logger.info(f"Generating {num_random_vectors} fixed random vectors...")
        self.fixed_vectors = fixed_vectors
        # self.fixed_vectors = self.ps.get_bound_vector()
        # iter_num = 0
        # while True:
        #     vector = self.ps.random_vector(can_perple = False)
        #     if vector not in self.fixed_vectors:
        #         self.fixed_vectors.append(vector)
        #         iter_num += 1
        #     if iter_num == num_random_vectors:
        #         break
        # 2. 生成所有任务组合 (Litmus x Vector)
        self.todo_queue = []
        for litmus in litmus_list:
            for vec in self.fixed_vectors:
                self.todo_queue.append((litmus, vec))

        # 打乱顺序，避免同一个 litmus 连续跑导致板子过热或其他偏差（可选）
        # random.shuffle(self.todo_queue)

        self.logger.info(
            f"Total tasks generated: {len(self.todo_queue)} (Litmus files: {len(litmus_list)} * Vectors: {num_random_vectors})")

        self.cache = ResultCache(stat_log + ".cache.jsonl")

        # 初始化流水线
        self.logger.info("Initializing Async Pipeline...")
        self.pipeline = LitmusPipeline(
            host=pipeline_host,
            port=pipeline_port,
            username=pipeline_user,
            password=pipeline_pass,
            logger = self.logger,
            resource_manager=self.resource_manager,
            remote_work_dir=remote_path
        )
        self.pipeline.start(compiler_thread_count=2, downloader_thread_count=2)

    def _submit_one(self, litmus, vec):
        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()
        # 把 vector 挂载到 params 上，方便结果回来时找回
        params._temp_vec = vec

        litmus_file = f"{litmus_path}/{litmus}.litmus"

        self.pipeline.submit_task(
            litmus_path=litmus_file,
            params=params,
            litmus_dir_path=litmus_dir_path,
            log_dir_path=self.log_dir,
            run_time=100000
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
                self._submit_one(litmus, vec)
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


# ================= 配置路径 =================
# litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
# stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_random.log"
# dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
# log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat_random.csv"
# litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/bayes'

litmus_path = "/home/software/桌面/bayes/perple_test_riscv/all_allow_litmus_C910_naive"
stat_log = "/home/software/桌面/bayes/perple_test_riscv/bayes_stat/"
dir_path = "/home/software/桌面/bayes/perple_test_riscv/bayes_log"
log_path = "/home/software/桌面/bayes/perple_test_riscv/bayes_stat/log_stat_bayes.csv"
litmus_dir_path = '/home/software/桌面/bayes/perple_test_riscv/bayes'

# litmus_vec_path="/home/software/桌面/bayes/perple_test_riscv/bayes_stat/litmus_vector.log"


# SSH 配置
# host = "198.168.226.168"
host = "10.42.0.131"
port = 22
username = "sipeed"
password = "sipeed"
remote_path = "/home/sipeed/test"
BOARDS = [
    {"host": "10.42.0.28", "port": 22, "user": "sipeed", "pass": "licheepi"},
    {"host": "10.42.0.46", "port": 22, "user": "sipeed", "pass": "licheepi"},
    {"host": "10.42.0.48", "port": 22, "user": "sipeed", "pass": "licheepi"},
    {"host": "10.42.0.58", "port": 22, "user": "sipeed", "pass": "licheepi"},
    {"host": "10.42.0.61", "port": 22, "user": "sipeed", "pass": "licheepi"},
    {"host": "10.42.0.112", "port": 22, "user": "sipeed", "pass": "licheepi"},
    {"host": "10.42.0.139", "port": 22, "user": "sipeed", "pass": "licheepi"},
    {"host": "10.42.0.228", "port": 22, "user": "sipeed", "pass": "licheepi"},
]


def launch_board_runner(board_config, litmus_list, param_space, global_manager, shared_vectors):
    host = board_config["host"]
    port = board_config["port"]
    username = board_config["user"]
    password = board_config["pass"]
    my_stat_log = f"{stat_log}/log_record_{host}.log"

    runner = RandomGridRunner(
        litmus_list,
        param_space,
        my_stat_log,
        resource_manager=global_manager,
        fixed_vectors = shared_vectors,
        num_random_vectors=50,  # <--- 这里控制向量数量
        pipeline_host=host,
        pipeline_user=username,
        pipeline_pass=password,
        pipeline_port=port
    )

    runner.run()

if __name__ == "__main__":
    random.seed(SEED)

    # 初始化 Log
    ts = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        log_file=f"{stat_log}.{ts}.run.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"Start Random Grid Run | seed={SEED}")

    # 获取 Litmus 列表
    litmus_list = get_files(litmus_path)

    # 提取文件名（去掉路径和后缀）
    litmus_list = [path.split("/")[-1][:-7] for path in litmus_list]

    logger.info(f"Loaded {len(litmus_list)} litmus tests.")

    param_space = LitmusParamSpace()
    threads = []
    global_fixed_vectors = param_space.get_bound_vector()
    iter_num = 0
    while True:
        vector = param_space.random_vector(can_perple=False)
        if vector not in global_fixed_vectors:
            global_fixed_vectors.append(vector)
            iter_num += 1
        if iter_num == 50:  # num_random_vectors
            break
    num_boards = len(BOARDS)
    global_manager = SharedResourceManager(total_consumers=num_boards)

    for board in BOARDS:
        t = threading.Thread(
            target=launch_board_runner,
            args=(board, litmus_list, param_space, global_manager, global_fixed_vectors)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()