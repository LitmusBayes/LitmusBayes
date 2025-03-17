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


# ================= 新的核心运行类：验证推荐参数 =================
class BestParamRunner(LitmusRunner):
    def __init__(
            self,
            recommendation_json_path,  # 新增：推荐结果的JSON路径
            param_space,
            stat_log,
            logger = None,
            mode="time",
            pipeline_host="192.168.1.105",
            pipeline_user="root",
            pipeline_pass="riscv",
            pipeline_port=22
    ):
        # 初始化父类，litmus_list 暂时传空，我们在下面自己解析
        super().__init__([], [], stat_log, mode)
        self.ps = param_space
        self.logger = get_logger(LOG_NAME)

        # 1. 加载推荐结果 JSON
        self.logger.info(f"Loading recommendations from {recommendation_json_path}...")
        if not os.path.exists(recommendation_json_path):
            raise FileNotFoundError(f"JSON file not found: {recommendation_json_path}")

        with open(recommendation_json_path, 'r') as f:
            self.recommendations = json.load(f)

        # 2. 构建任务队列 (Litmus, Vector, PredictedScore)
        self.todo_queue = []

        # 遍历 JSON 构建任务
        for litmus_name, info in self.recommendations.items():
            vec = info['param']
            pred_score = info.get('pred_score', -1.0)  # 获取预测分用于对比，没有则为-1

            # 检查文件是否存在（防止 json 里有文件名但磁盘上没有）
            full_path = f"{litmus_path}/{litmus_name}.litmus"
            if os.path.exists(full_path):
                self.todo_queue.append({
                    "litmus": litmus_name,
                    "vec": vec,
                    "pred": pred_score
                })
            else:
                self.logger.warning(f"File not found on disk, skipping: {litmus_name}")

        # 可选：打乱顺序（虽然是验证，但打乱有助于均衡负载）
        random.shuffle(self.todo_queue)

        self.logger.info(f"Total validation tasks loaded: {len(self.todo_queue)}")

        # 3. 初始化 Cache 和 Pipeline
        self.cache = ResultCache(stat_log + ".validation_cache.jsonl")

        self.logger.info("Initializing Async Pipeline...")
        self.pipeline = LitmusPipeline(
            logger = self.logger,
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

        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()

        # 把 vector 和 预测分 挂载到 params 上，方便结果回来时找回
        params._temp_vec = vec
        params._temp_pred = pred

        litmus_file = f"{litmus_path}/{litmus}.litmus"

        self.pipeline.submit_task(
            litmus_path=litmus_file,
            params=params,

            litmus_dir_path=litmus_dir_path,
            log_dir_path=dir_path,
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
        self.logger.info("=== Starting Validation Execution Loop ===")

        MAX_IN_FLIGHT = 16
        active_count = 0
        finished_count = 0
        skipped_count = 0

        # 辅助函数：填充流水线
        def try_fill_pipeline():
            nonlocal active_count
            while len(self.todo_queue) > 0 and active_count < MAX_IN_FLIGHT:
                task_info = self.todo_queue.pop(0)  # 取出一个 dict

                # Cache Check
                cached_score = self.cache.get(task_info['litmus'], task_info['vec'])
                if cached_score is not None:
                    # 如果缓存里有，我们可以打个日志对比一下之前的分数和预测的分数
                    # self.logger.info(f"[SKIP-CACHED] {task_info['litmus']} | Actual: {cached_score:.4f} vs Pred: {task_info['pred']:.4f}")
                    nonlocal skipped_count
                    skipped_count += 1
                    continue

                # Submit
                self._submit_one(task_info)
                active_count += 1

        # --- 1. 初始启动 ---
        try_fill_pipeline()

        if active_count == 0 and len(self.todo_queue) == 0:
            self.logger.info("Nothing to run (All cached or empty queue).")
            # 即使直接退出，也建议先把 pipeline 停掉
            self.pipeline.wait_completion()
            return

        # --- 2. 事件循环 ---
        for result in self.pipeline.stream_results():
            active_count -= 1
            finished_count += 1

            task = result['task']
            log_path = result['log_path']
            litmus_name = task.litmus_path.split("/")[-1][:-7]

            if hasattr(task.params, '_temp_vec'):
                param_vec = task.params._temp_vec
                pred_score = getattr(task.params, '_temp_pred', 0.0)

                # 算分 (Actual Score)
                actual_score = self._parse_log_to_score(log_path, litmus_name, task.params.is_perple(), self.mode)

                # 计算偏差
                diff = actual_score - pred_score

                self.logger.info(
                    f"[DONE] {litmus_name:<20} | Act: {actual_score:.4f} | Pred: {pred_score:.4f} | Diff: {diff:+.4f} | Rem: {len(self.todo_queue)}")

                # 存 Cache
                self.cache.add(litmus_name, param_vec, actual_score)
            else:
                self.logger.error("Lost vector info in task result!")

            try_fill_pipeline()

            if len(self.todo_queue) == 0 and active_count == 0:
                self.logger.info("Queue empty and Pipeline drained. Exiting loop.")
                break

        self.logger.info(f"Validation Finished. Processed: {finished_count}, Skipped: {skipped_count}")
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
stat_log = "./log_record_final_1.log"
dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat_random.csv"
litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/bayes'

json_recommendation_path = "best_params_recommendation_robust_final.json"

# SSH 配置
# host = "10.42.0.58"
host = "10.42.0.131"
port = 22
# username = "root"
# password = "bianbu"
# remote_path = "/root/test"
username = "sipeed"
password = "sipeed"
remote_path = "/home/sipeed/test"

def get_env_var(var_name, is_int=False):
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"❌ 错误: 缺少必需的环境变量 '{var_name}'")

    if is_int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"❌ 错误: 环境变量 '{var_name}' 必须是数字，当前值为: '{value}'")

    return value


host = get_env_var("C910_HOST")
port = get_env_var("C910_PORT", is_int=True)
username = get_env_var("C910_USERNAME")
password = get_env_var("C910_PASSWORD")
remote_path = get_env_var("C910_REMOTE_PATH")
if __name__ == "__main__":
    random.seed(SEED)
    stat_logs = [
        "./final_result/bayes_final/log_record_final_1.log",
        "./final_result/bayes_final/log_record_final_2.log",
        "./final_result/bayes_final/log_record_final_3.log",
        "./final_result/bayes_final/log_record_final_4.log",
        "./final_result/bayes_final/log_record_final_5.log",
        "./final_result/bayes_final/log_record_final_6.log",
        "./final_result/bayes_final/log_record_final_7.log",
        "./final_result/bayes_final/log_record_final_8.log",
    ]

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
    for idx, current_stat_log in enumerate(stat_logs, 1):

        # 实例化运行器
        runner = BestParamRunner(
            recommendation_json_path=json_recommendation_path,  # 传入 JSON 路径
            param_space=param_space,
            stat_log=current_stat_log,
            logger = logger,
            pipeline_host=host,
            pipeline_user=username,
            pipeline_pass=password,
            pipeline_port=port
        )

        # 开始运行
        runner.run()

        # 可选：给服务器或远端机器预留一点点关闭旧 Pipeline 进程的缓冲时间
        time.sleep(20)