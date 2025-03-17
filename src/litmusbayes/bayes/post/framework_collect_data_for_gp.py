import json
import logging
import os
import random
import time
from collections import defaultdict

from src.litmusbayes.bayes.LitmusPipeline import LitmusPipeline
from src.litmusbayes.bayes.framework import LitmusRunner
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files, parse_log_by_mode_perple, parse_log_by_mode

SEED = 2025
LOG_NAME = "representative_grid"  # 修改日志名称


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


# ================= 新的核心运行类 =================
class RepresentativeGridRunner(LitmusRunner):
    def __init__(
            self,
            cluster_json_path,  # 聚类结果路径
            params_json_path,  # 筛选参数路径
            param_space,
            stat_log,
            logger=None,
            mode="time",
            pipeline_host="192.168.1.105",
            pipeline_user="root",
            pipeline_pass="riscv",
            pipeline_port=22
    ):
        # 注意：父类初始化不再需要完整的 litmus_list，我们传个空列表也没事，因为我们在下面自己构建
        super().__init__([], [], stat_log, mode)
        self.ps = param_space
        self.logger = get_logger(LOG_NAME)

        # 1. 加载聚类中心 (Representatives)
        self.target_litmus_list = self._load_representatives(cluster_json_path)

        # 2. 加载筛选后的参数 (Refined Vectors)
        self.target_vectors = self._load_params(params_json_path)

        # 3. 生成任务组合 (Representative x Refined Vector)
        #
        self.todo_queue = []
        for litmus in self.target_litmus_list:
            for vec in self.target_vectors:
                self.todo_queue.append((litmus, vec))

        # 打乱顺序，避免同一种参数连续跑导致过热
        random.shuffle(self.todo_queue)

        self.logger.info("=== Configuration Loaded ===")
        self.logger.info(f"Target Tests (Centers): {len(self.target_litmus_list)}")
        self.logger.info(f"Target Params (Refined): {len(self.target_vectors)}")
        self.logger.info(f"Total Tasks: {len(self.todo_queue)}")

        self.cache = ResultCache(stat_log + ".cache.jsonl")

        # 初始化流水线
        self.logger.info("Initializing Async Pipeline...")
        self.pipeline = LitmusPipeline(
            logger=self.logger,
            host=pipeline_host,
            port=pipeline_port,
            username=pipeline_user,
            password=pipeline_pass,
            remote_work_dir=remote_path
        )
        self.pipeline.start()

    def _load_representatives(self, path):
        """解析聚类 JSON，提取 representative 字段"""
        self.logger.info(f"Loading cluster centers from {path}...")
        targets = []
        with open(path, 'r') as f:
            data = json.load(f)
            # data 格式: {"0": {"representative": "Name", ...}, ...}
            for cluster_id, info in data.items():
                if "representative" in info:
                    targets.append(info["representative"])

        # 去重（以防万一）
        targets = list(set(targets))
        return targets

    def _load_params(self, path):
        """解析参数 JSON"""
        self.logger.info(f"Loading refined params from {path}...")
        with open(path, 'r') as f:
            vectors = json.load(f)
        return vectors

    def _submit_one(self, litmus, vec):
        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()
        params._temp_vec = vec

        # 注意：这里需要 litmus_path 全局变量或者传进来，确保能找到文件
        litmus_file = f"{litmus_path}/{litmus}.litmus"

        self.pipeline.submit_task(
            litmus_path=litmus_file,
            params=params,
            litmus_dir_path=litmus_dir_path,
            log_dir_path=dir_path,
            run_time=10000
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
        self.logger.info("=== Starting Representative Grid Execution Loop ===")
        MAX_IN_FLIGHT = 16
        active_count = 0
        finished_count = 0
        skipped_count = 0

        def try_fill_pipeline():
            nonlocal active_count
            while len(self.todo_queue) > 0 and active_count < MAX_IN_FLIGHT:
                litmus, vec = self.todo_queue.pop(0)

                # Cache Check
                if self.cache.get(litmus, vec) is not None:
                    nonlocal skipped_count
                    skipped_count += 1
                    continue

                self._submit_one(litmus, vec)
                active_count += 1

        # 1. 初始启动
        try_fill_pipeline()

        if active_count == 0 and len(self.todo_queue) == 0:
            self.logger.info("Nothing to run (All cached or empty queue).")
            # 即使没任务也要关闭 pipeline
            self.pipeline.wait_completion()
            return

        # 2. 事件循环
        for result in self.pipeline.stream_results():
            active_count -= 1
            finished_count += 1

            task = result['task']
            log_path = result['log_path']
            litmus_name = task.litmus_path.split("/")[-1][:-7]

            if hasattr(task.params, '_temp_vec'):
                param_vec = task.params._temp_vec
                score = self._parse_log_to_score(log_path, litmus_name, task.params.is_perple(), self.mode)

                # 记录详细日志方便 grep
                self.logger.info(
                    f"[DONE] {litmus_name} | Param: {param_vec[:3]}... | Score: {score:.4f} | Rem: {len(self.todo_queue)}")

                self.cache.add(litmus_name, param_vec, score)
            else:
                self.logger.error("Lost vector info in task result!")

            try_fill_pipeline()

            if len(self.todo_queue) == 0 and active_count == 0:
                break

        self.logger.info(f"Run Finished. Total Processed: {finished_count}, Skipped: {skipped_count}")
        self.pipeline.wait_completion()


# ================= 配置路径 =================

# 1. Litmus 原始文件所在目录 (用于 Pipeline 读取物理文件)
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"

# 2. 两个关键的 JSON 输入文件
# 聚类结果文件 (包含 representatives)
cluster_json_path = "./cluster_results/cluster_centers.json"
# 筛选后的参数文件 (包含 vectors)
refined_params_path = "./cluster_results/refined_configs.json"

# 3. 输出日志配置
stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_representative_run.log"
dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log_banana_gp"
litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/bayes'

# SSH 配置
host = "10.42.0.58"
port = 22
username = "root"
password = "bianbu"
remote_path = "/root/test"

if __name__ == "__main__":
    random.seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        log_file=f"{stat_log}.{ts}.run.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"Start Representative Grid Run | seed={SEED}")

    # 检查文件是否存在
    if not os.path.exists(cluster_json_path):
        logger.error(f"Cluster file not found: {cluster_json_path}")
        exit(1)
    if not os.path.exists(refined_params_path):
        logger.error(f"Params file not found: {refined_params_path}")
        exit(1)

    param_space = LitmusParamSpace()

    # 实例化新的运行器
    runner = RepresentativeGridRunner(
        cluster_json_path=cluster_json_path,
        params_json_path=refined_params_path,
        param_space=param_space,
        stat_log=stat_log,
        logger=logger,
        pipeline_host=host,
        pipeline_user=username,
        pipeline_pass=password,
        pipeline_port=port
    )

    runner.run()