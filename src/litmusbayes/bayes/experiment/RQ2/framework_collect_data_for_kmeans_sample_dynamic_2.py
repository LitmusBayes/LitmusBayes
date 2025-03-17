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
LOG_NAME = "validation_run"


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


class BestParamRunner(LitmusRunner):
    def __init__(
            self,
            recommendation_json_path,
            param_space,
            stat_log,
            logger=None,
            mode="time",
            pipeline_host="192.168.1.105",
            pipeline_user="root",
            pipeline_pass="riscv",
            pipeline_port=22
    ):
        super().__init__([], [], stat_log, mode)
        self.ps = param_space
        self.logger = logger if logger else get_logger(LOG_NAME)

        # 定义探测用的初始参数
        self.probe_vec = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

        # 1. 加载新的聚类 JSON 格式
        self.logger.info(f"Loading cluster recommendations from {recommendation_json_path}...")
        if not os.path.exists(recommendation_json_path):
            raise FileNotFoundError(f"JSON file not found: {recommendation_json_path}")

        with open(recommendation_json_path, 'r') as f:
            self.clusters = json.load(f)

        self.todo_queue = []
        self.cluster_states = {}  # 用于记录每个 cluster 测到第几个 member 了

        # 2. 初始化任务队列，将每个 cluster 的第一个 member 作为 probe 任务推入
        for cluster_id, info in self.clusters.items():
            members = info.get("members", [])
            if not members:
                continue

            # 记录 cluster 的状态
            self.cluster_states[cluster_id] = {
                "members": members,
                "current_idx": 0,
                "success_count": 0  # 新增：记录当前聚类已经成功采样的数量
            }

            first_member = members[0]
            # 这里的 litmus 名字可能需要像你之前那样做切片处理，视你的实际文件名而定
            # 如果不需要切片，请直接用 first_member
            litmus_name = "_".join(first_member.split("_")[:-2]) if "_" in first_member else first_member

            # 压入初始探测任务
            self.todo_queue.append({
                "litmus": litmus_name,
                "vec": self.probe_vec,
                "pred": 1,
                "task_type": "probe",  # 标记为探测任务
                "cluster_id": cluster_id,
                "original_member_name": first_member  # 保留原始名字以便后续找回
            })

        # 打乱顺序，均衡流水线负载
        random.shuffle(self.todo_queue)
        self.logger.info(f"Total clusters loaded: {len(self.cluster_states)}")

        # 3. 初始化 Cache 和 Pipeline
        self.cache = ResultCache(stat_log + ".validation_cache.jsonl")
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

    def _submit_one(self, task_info):
        litmus = task_info['litmus']
        vec = task_info['vec']
        pred = task_info['pred']
        task_type = task_info.get('task_type', 'full_suite')
        cluster_id = task_info.get('cluster_id')
        original_member = task_info.get('original_member_name')

        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()

        # 把任务的元数据挂载到 params 上，结果回来时依靠它们做分支判断
        params._temp_vec = vec
        params._temp_pred = pred
        params._temp_task_type = task_type
        params._temp_cluster_id = cluster_id
        params._temp_litmus = litmus
        params._temp_original_member = original_member

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

        # --- 核心逻辑：根据分数判断下一步做什么 ---
    def _handle_logic_after_score(self, task_info, actual_score):
        task_type = task_info.get('task_type')
        cluster_id = task_info.get('cluster_id')
        litmus_name = task_info['litmus']

        # 获取当前聚类的状态字典
        state = self.cluster_states.get(cluster_id)
        if not state:
            return

        if task_type == "probe":
            # 分数不为 0 且不为 -1 (解析错误) 视为探测成功
            if actual_score != 0 and actual_score != -1:
                self.logger.info(
                    f"[PROBE SUCCESS] Cluster {cluster_id} | Member: {litmus_name} | Score: {actual_score:.4f}. Queuing full suite.")

                # 记录该聚类成功收集的样本数
                state["success_count"] += 1

                # 探测成功，将这组 test 进行所有重要参数的运行
                vec_list = self.ps.get_vector_by_important()
                for vec in vec_list:
                    self.todo_queue.append({
                        "litmus": litmus_name,
                        "vec": vec,
                        "pred": 1,
                        "task_type": "full_suite",  # 标记为全量测试，避免无限循环
                        "cluster_id": cluster_id
                    })
            else:
                self.logger.info(
                    f"[PROBE FAILED] Cluster {cluster_id} | Member: {litmus_name} | Score: {actual_score}. Trying next member...")

            # 关键修改：无论刚才是成功还是失败，只要当前聚类收集成功的样本数不足 2 个，就继续找下一个 member 进行探测
            if state["success_count"] < 2:
                state["current_idx"] += 1

                if state["current_idx"] < len(state["members"]):
                    next_member = state["members"][state["current_idx"]]
                    next_litmus_name = "_".join(next_member.split("_")[:-2]) if "_" in next_member else next_member

                    self.todo_queue.append({
                        "litmus": next_litmus_name,
                        "vec": self.probe_vec,
                        "pred": 1,
                        "task_type": "probe",
                        "cluster_id": cluster_id,
                        "original_member_name": next_member
                    })
                else:
                    self.logger.warning(
                        f"[CLUSTER EXHAUSTED] Cluster {cluster_id} has no valid members left. Only found {state['success_count']} valid samples.")
            elif actual_score != 0 and actual_score != -1:
                # 只有当正好达成 2 个且本次是成功的时候，打印满足条件的日志，避免重复打印
                self.logger.info(f"[CLUSTER SATISFIED] Cluster {cluster_id} has successfully collected 2 samples.")
    def run(self):
        self.logger.info("=== Starting Validation Execution Loop ===")

        MAX_IN_FLIGHT = 16
        active_count = 0
        finished_count = 0
        skipped_count = 0

        # 辅助函数：填充流水线
        def try_fill_pipeline():
            nonlocal active_count, skipped_count
            while len(self.todo_queue) > 0 and active_count < MAX_IN_FLIGHT:
                task_info = self.todo_queue.pop(0)

                # Cache Check
                cached_score = self.cache.get(task_info['litmus'], task_info['vec'])
                if cached_score is not None:
                    skipped_count += 1
                    # 如果命中了缓存，直接跳过运行，但【必须】执行后续的条件判断逻辑！
                    self._handle_logic_after_score(task_info, cached_score)
                    continue

                # Submit
                self._submit_one(task_info)
                active_count += 1

        # 1. 初始启动
        try_fill_pipeline()

        if active_count == 0 and len(self.todo_queue) == 0:
            self.logger.info("Nothing to run (All cached or empty queue).")
            self.pipeline.wait_completion()
            return

        # 2. 事件循环
        for result in self.pipeline.stream_results():
            active_count -= 1
            finished_count += 1

            task = result['task']
            log_path = result['log_path']

            if hasattr(task.params, '_temp_vec'):
                param_vec = task.params._temp_vec
                pred_score = getattr(task.params, '_temp_pred', 0.0)
                litmus_name = getattr(task.params, '_temp_litmus', "UNKNOWN")

                # 计算实际得分
                actual_score = self._parse_log_to_score(log_path, litmus_name, task.params.is_perple(), self.mode)
                diff = actual_score - pred_score

                task_type_str = "PROBE" if getattr(task.params, '_temp_task_type') == "probe" else "SUITE"
                self.logger.info(
                    f"[DONE-{task_type_str}] {litmus_name:<20} | Act: {actual_score:.4f} | Diff: {diff:+.4f} | Rem in Queue: {len(self.todo_queue)}")

                # 存 Cache
                self.cache.add(litmus_name, param_vec, actual_score)

                # 重构一个 task_info 字典传递给逻辑判断器
                reconstructed_task_info = {
                    "litmus": litmus_name,
                    "vec": param_vec,
                    "task_type": getattr(task.params, '_temp_task_type'),
                    "cluster_id": getattr(task.params, '_temp_cluster_id')
                }

                # 执行分支判断 (决定是测试下一个 member 还是跑满所有 vec)
                self._handle_logic_after_score(reconstructed_task_info, actual_score)

            else:
                self.logger.error("Lost vector info in task result!")

            # 只要有任务或者有空间，继续填满 Pipeline
            try_fill_pipeline()

            if len(self.todo_queue) == 0 and active_count == 0:
                self.logger.info("Queue empty and Pipeline drained. Exiting loop.")
                break

        self.logger.info(f"Validation Finished. Processed: {finished_count}, Skipped (Cached): {skipped_count}")
        self.pipeline.wait_completion()

# ================= 配置路径 =================
# litmus_path = "./make_new_litmus/litmus_output"
# stat_log = "./make_new_litmus/log_record_random.log"
# dir_path = "./make_new_litmus/bayes_log"
# log_path = "./make_new_litmus/log_stat_random.csv"
# litmus_dir_path = './make_new_litmus/litmus_output_bayes'
# 指向你刚才生成的 JSON 文件
# json_recommendation_path = "best_params_recommendation1.json"
# litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
# stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_random.log"
# dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
# log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat_random.csv"
# litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/bayes'
#
# json_recommendation_path = "./best_params_recommendation_robust.json"


litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log = "./log_record_init_banana_important_kmeans_sample_2_10.log"
dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log_banana_final"
log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat_random.csv"
litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/bayes'

json_recommendation_path = "cluster_results_final_10/cluster_centers.json"

# SSH 配置
host = "10.42.0.58"
# host = "10.42.0.131"
port = 22
username = "root"
password = "bianbu"
remote_path = "/root/test"
# username = "sipeed"
# password = "sipeed"
# remote_path = "/home/sipeed/test"


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
    logger.info(f"Start Validation Run | JSON source: {json_recommendation_path}")

    param_space = LitmusParamSpace()

    # 实例化运行器
    runner = BestParamRunner(
        recommendation_json_path=json_recommendation_path,  # 传入 JSON 路径
        param_space=param_space,
        stat_log=stat_log,
        logger = logger,
        pipeline_host=host,
        pipeline_user=username,
        pipeline_pass=password,
        pipeline_port=port
    )

    # 开始运行
    runner.run()