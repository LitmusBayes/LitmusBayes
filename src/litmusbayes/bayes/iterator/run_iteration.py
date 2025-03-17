import json
import logging
import os
import glob
import time
import numpy as np
from src.litmusbayes.bayes.iterator import data_collector, model_optimizer
from src.litmusbayes.bayes.logger_util import setup_logger

# ================= 配置区域 =================
ITERATIONS = 40  # 迭代总轮数

# 路径配置
BASE_DIR = "/home/software/桌面/bayes/perple_test_riscv"
# BASE_DIR = "/home/whq/Desktop/code_list/perple_test/"

# LITMUS_PATH = f"{BASE_DIR}/all_allow_litmus_C910_naive"
LITMUS_PATH = f"../selected_litmus_tests"
STAT_LOG_DIR = f"{BASE_DIR}/bayes_stat"
LITMUS_VEC_PATH = f"{STAT_LOG_DIR}/litmus_vector.log"

# 关键文件
BASELINE_SCORE_FILE = f"../log/baseline_scores_0.json"
# GLOBAL_CACHE_FILE = f"{STAT_LOG_DIR}/log_record_bayes.log.cache4_norm.jsonl"
GLOBAL_CACHE_FILE = f"../log/cache_norm.jsonl"
RECOMMEND_FILE = "best_params_recommendation_robust.json"

# 板子配置
REMOTE_WORK_DIR = "/home/sipeed/test"
LITMUS_DIR_ON_PIPELINE = f"{BASE_DIR}/bayes"
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

logger = setup_logger(
    log_file="./main_iteration.log",
    level=logging.INFO,
    name="MainLoop",
    stdout=True
)


def load_baseline_scores(path):
    if not os.path.exists(path):
        logger.error(f"Baseline file {path} not found!")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def process_and_merge_logs(board_logs_dir, baseline_scores, output_cache_path, iter):
    """
    1. 扫描所有 board 的 cache 文件
    2. 聚合 (Litmus, Param) -> [scores...]
    3. 过滤 -1，求平均
    4. 归一化 (Avg / Baseline)
    5. 追加写入到 Global Cache (output_cache_path)
    """
    logger.info(">>> Start merging and normalizing logs...")

    # 临时字典: key -> list of valid scores
    temp_data = {}  # key: "litmus|p1,p2..."

    # 1. 查找所有板子的 cache 文件
    files = []
    for board in BOARDS:
        files.append(f"{board_logs_dir}/log_record_{board['host']}-{iter}.log.cache.jsonl")
    # pattern = f"{board_logs_dir}/log_record_*.log.cache.jsonl"
    # files = glob.glob(pattern)
    logger.info(f"Found {len(files)} board cache files.")

    new_records_count = 0

    for fname in files:
        with open(fname, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    score = obj["score"]
                    # 过滤无效分数
                    if score < 0: continue

                    # 构造 key
                    key = f"{obj['litmus']}|" + ",".join(map(str, obj['param']))
                    if key not in temp_data:
                        temp_data[key] = []
                    temp_data[key].append(score)
                except:
                    continue

    # 2. 计算平均值并归一化
    # 我们只关心这次新跑出来的数据，还是说把所有历史重新算一遍？
    # 通常为了性能，我们这里只处理这一轮产生的增量，但是 board cache 文件如果不清空，它是累积的。
    # 为了简单且准确，我们读取所有 board cache，计算出结果，然后去重写入 Global Cache。
    # 这里的 Global Cache 是只追加模式，所以我们需要判断是否已经写过。
    # 简单的做法：读取 Global Cache 所有 key，如果不包含当前 key，则写入。

    existing_keys = set()
    if os.path.exists(output_cache_path):
        with open(output_cache_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    k = f"{obj['litmus']}|" + ",".join(map(str, obj['param']))
                    existing_keys.add(k)
                except:
                    pass

    with open(output_cache_path, "a") as f_out:
        for key, scores in temp_data.items():
            if key in existing_keys:
                continue  # 已存在，跳过

            # 分离 key 里的信息
            litmus_name, param_str = key.split("|")
            param_vec = [int(x) for x in param_str.split(",")]

            # 计算平均分
            avg_score = np.mean(scores)

            # 归一化
            baseline = baseline_scores.get(litmus_name, -1)
            if baseline is None or baseline <= 0:
                logger.warning(f"No baseline for {litmus_name}, skipping normalization (using raw).")
                norm_score = avg_score  # 或者设为 0
            else:
                norm_score = avg_score / baseline

            # 写入
            record = {
                "litmus": litmus_name,
                "param": param_vec,
                "score": norm_score
            }
            f_out.write(json.dumps(record) + "\n")
            new_records_count += 1
            existing_keys.add(key)  # 更新当前内存中的key

    logger.info(f"<<< Merge done. Added {new_records_count} new normalized records.")


def main():
    # 0. 准备 Baseline
    baseline = load_baseline_scores(BASELINE_SCORE_FILE)
    if not baseline:
        logger.error("Cannot proceed without baseline scores.")
        return

    for i in range(ITERATIONS):
        logger.info(f"\n{'=' * 20} Iteration {i + 1}/{ITERATIONS} {'=' * 20}")

        # === Step 1: Model Optimization & Recommendation ===
        logger.info("Running Optimizer...")
        recommendations = model_optimizer.main_optimizer(
            litmus_path=LITMUS_PATH,
            cache_file_path=GLOBAL_CACHE_FILE,  # 读取归一化后的数据
            litmus_vec_path=LITMUS_VEC_PATH,
            output_json_path=RECOMMEND_FILE,
            log_base_path=f"{STAT_LOG_DIR}/iter_{i}"
        )

        # === Step 2: Filter Tasks ===
        # 提取需要跑的任务 (排除 is_cached=True 的)
        tasks_to_run = []
        for litmus, info in recommendations.items():
            if not info['is_cached']:
                tasks_to_run.append({
                    "litmus": litmus,
                    "param": info['param']
                })
            else:
                # 可以在这里记录一下跳过了多少
                pass

        logger.info(f"Tasks generated: {len(recommendations)}. After filtering cached: {len(tasks_to_run)}")

        if not tasks_to_run:
            logger.info("No new tasks to run. Optimization converged or cache full.")
            break

        # === Step 3: Data Collection ===
        logger.info("Running Data Collector...")
        collector_config = {
            "litmus_path_base": LITMUS_PATH,
            "stat_log_dir": STAT_LOG_DIR,
            "litmus_dir_path": LITMUS_DIR_ON_PIPELINE,
            "remote_work_dir": REMOTE_WORK_DIR
        }

        data_collector.main_collector(
            task_list=[item for item in tasks_to_run],
            board_configs=BOARDS,
            base_config=collector_config,
            iter = i
        )

        # === Step 4: Merge & Normalize Results ===
        # 将本轮各个板子的运行结果合并，归一化，并写入 GLOBAL_CACHE_FILE
        process_and_merge_logs(
            board_logs_dir=STAT_LOG_DIR,
            baseline_scores=baseline,
            output_cache_path=GLOBAL_CACHE_FILE,
            iter = i
        )

        # 可选：每轮结束后清空板子的 cache 文件？
        # 如果不清空，process_and_merge_logs 会重复读取，但依靠 existing_keys 去重
        # 为了性能，建议定期归档板子的 log，或者 process 逻辑里只读最近的文件

        time.sleep(5)  # 休息一下


if __name__ == "__main__":
    main()