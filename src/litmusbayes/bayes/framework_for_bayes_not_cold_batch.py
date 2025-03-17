import json
import logging
import os
import random
import time
from collections import defaultdict

from scipy.stats import norm

from src.litmusbayes.bayes.LitmusPipeline import LitmusPipeline
from src.litmusbayes.bayes.framework import LitmusRunner, get_score
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.litmus_params import LitmusParams
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files, read_log_to_summary, parse_log_by_mode_perple, parse_log_by_mode
import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np

SEED = 2025
LOG_NAME = "bayes"

def load_litmus_dict(json_file_path):
    """
    读取生成的 JSON 文件并返回字典
    """
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在 {json_file_path}")
        return {}

    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}

        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    key = self._make_key(obj["litmus"], obj["param"])
                    self.data[key] = obj["score"]

        # 追加写
        self.f = open(path, "a")

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        key = self._make_key(litmus, param_vec)
        if key in self.data:
            return

        self.data[key] = score
        self.f.write(json.dumps({
            "litmus": litmus,
            "param": param_vec,
            "score": score
        }) + "\n")
        self.f.flush()

class ErrorCache:
    """
    Cache for failed / invalid litmus runs.
    Once a (litmus, param) is recorded here, it should never be retried.
    """

    def __init__(self, path):
        self.path = path
        self.data = {}

        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    key = self._make_key(obj["litmus"], obj["param"])
                    self.data[key] = obj

        # append-only
        self.f = open(path, "a")

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def has(self, litmus, param_vec):
        """
        Check whether this (litmus, param) has failed before.
        """
        return self._make_key(litmus, param_vec) in self.data

    def get(self, litmus, param_vec):
        """
        Get error record (dict) or None.
        """
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, error_type, error_msg=None, extra=None):
        """
        Record a failed run.

        Args:
            error_type: short string, e.g. "timeout", "compile_error", "runtime_error"
            error_msg: optional detailed message
            extra: optional dict for additional info
        """
        key = self._make_key(litmus, param_vec)
        if key in self.data:
            return

        record = {
            "litmus": litmus,
            "param": param_vec,
            "error_type": error_type,
            "error_msg": error_msg,
            "extra": extra,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.data[key] = record
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()


class RandomForestBO:

    def __init__(self, param_space: LitmusParamSpace, litmus_list, litmus_vec_path, can_perple_list, n_estimators=200):
        self.ps = param_space
        self.can_perple_list = can_perple_list
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=1,
            max_features="sqrt",
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.max_litmus_score = {}
        self.litmus_to_vector_dict = {}
        self.litmus_list = litmus_list
        self.litmus_run_count = defaultdict(int)
        self.max_runs_per_litmus = 150

        # 用于记录期望的特征总长度，第一次 add 时确定
        self.expected_dim = None

        # ===== 关键：加载 litmus 向量 =====
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

        # ===== 新增：检查 litmus 向量长度一致性 =====
        first_vec_len = None
        for name, vec in self.litmus_to_vector_dict.items():
            if first_vec_len is None:
                first_vec_len = len(vec)
            elif len(vec) != first_vec_len:
                # 如果发现向量长度不一致，抛出异常并打印名字
                raise ValueError(
                    f"Litmus Vector dimension mismatch! '{name}' has len {len(vec)}, expected {first_vec_len}")

        if first_vec_len:
            print(f"[Check Pass] All litmus vectors have length: {first_vec_len}")

        # ===== 一致性检查 =====
        missing_vecs = [l for l in self.litmus_list if l not in self.litmus_to_vector_dict]

        if missing_vecs:
            self.litmus_list = [l for l in self.litmus_list if l in self.litmus_to_vector_dict]

        for l in self.litmus_to_vector_dict:
            if l not in self.litmus_list:
                # 这里的逻辑稍微有点怪，通常不需要删 vector_dict 里的，不过照旧保留你的逻辑
                # raise ValueError(f"Missing vector for litmus: {l}")
                pass  # 建议去掉这个报错，vector_dict 多余没关系

        self.logger = get_logger(LOG_NAME)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                name, vec_str = line.split(":", 1)
                try:
                    vec = eval(vec_str)
                    litmus_to_vec[name] = list(vec)
                except:
                    continue
        return litmus_to_vec

    # 加入训练数据 [修改版：增加长度检查]
    def add(self, litmus_name, param_vec, score):
        self.logger.info(f"Adding Train {litmus_name} with {param_vec} with score {score}")

        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        combined_feature = list(param_vec) + list(litmus_vec)
        current_len = len(combined_feature)

        # 1. 第一次添加数据，锁定维度
        if self.expected_dim is None:
            self.expected_dim = current_len
            print(
                f"[Init] Setting feature dimension to {self.expected_dim} (Param: {len(param_vec)}, Litmus: {len(litmus_vec)})")

        # 2. 后续添加数据，检查维度
        elif current_len != self.expected_dim:
            self.logger.error(f"Dimension Mismatch! Expected {self.expected_dim}, got {current_len}")
            self.logger.error(
                f"Details -> Litmus: {litmus_name}, ParamLen: {len(param_vec)}, VecLen: {len(litmus_vec)}")
            # 这里可以选择跳过，或者报错。为了防止 crash，我们选择跳过不加入
            return

        self.X.append(combined_feature)
        self.y.append(score)
        self.max_litmus_score[litmus_name] = max(self.max_litmus_score.get(litmus_name, 0), score)

    # 训练 RF
    def fit(self):
        # 再次防御性检查，防止 self.X 为空
        if not self.X:
            return

        # 使用 try-except 捕获 numpy 转换错误，方便调试
        try:
            X_arr = np.array(self.X)
            y_arr = np.array(self.y)
            self.model.fit(X_arr, y_arr)
        except ValueError as e:
            self.logger.error(f"Numpy conversion failed: {e}")
            # 打印前几个样本的长度进行调试
            lengths = [len(x) for x in self.X[:5]]
            self.logger.error(f"Lengths of first 5 samples: {lengths}")
            raise e

    def compute_ucb(self, candidate_vecs, beta=1.5):
        if len(candidate_vecs) == 0:
            return []

        C = np.array(candidate_vecs)

        # 确保候选集维度也是对的
        if C.shape[1] != self.expected_dim:
            # 如果候选集生成逻辑有问题，这里也可能报错
            self.logger.warning(f"Candidate dimension mismatch. Model expects {self.expected_dim}, got {C.shape[1]}")

        preds = np.array([
            est.predict(C) for est in self.model.estimators_
        ])

        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

        return mu + beta * sigma

    # ... (其他方法 sample_litmus, generate_candidates 等保持不变) ...
    # 注意：generate_candidates 里生成 candidate 时也需要保证 list(p) + list(litmus_vector) 长度正确

    def sample_litmus(self, n_litmus):
        available = [
            l for l in self.litmus_list
            if self.litmus_run_count[l] < self.max_runs_per_litmus
        ]
        if not available:
            return []
        k = min(n_litmus, len(available))
        return random.sample(available, k)

    def generate_candidates(self, n_litmus=5, n_param=200):
        candidates = []
        litmus_names = self.sample_litmus(n_litmus)
        for litmus in litmus_names:
            litmus_vector = self.litmus_to_vector_dict[litmus]
            for _ in range(n_param):
                can_perple = True if litmus in self.can_perple_list else False
                p = self.ps.random_vector(can_perple=can_perple)

                # 验证生成的候选点维度
                full_vec = list(p) + list(litmus_vector)
                if self.expected_dim and len(full_vec) != self.expected_dim:
                    continue  # 跳过维度不对的候选点

                candidates.append((litmus, full_vec))
        return candidates

    def select_batch_next(self, batch_size=8):
        if len(self.X) == 0:
            return []
        self.fit()
        n_sample_litmus = max(batch_size * 2, 10)
        cands = self.generate_candidates(n_litmus=n_sample_litmus, n_param=1000)
        if not cands:
            return []

        all_vecs = [item[1] for item in cands]
        X_all = np.array(all_vecs)
        all_ucb_scores = self.compute_ucb(X_all, beta=1.96)

        scored_candidates = []
        for i, (litmus, vec) in enumerate(cands):
            scored_candidates.append({
                "litmus": litmus,
                "vec": vec,
                "ei": all_ucb_scores[i]
            })
        scored_candidates.sort(key=lambda x: x["ei"], reverse=True)
        best_batch = []
        seen_litmus = set()
        for item in scored_candidates:
            if len(best_batch) >= batch_size:
                break
            litmus_name = item["litmus"]
            if litmus_name in seen_litmus:
                continue
            best_batch.append((litmus_name, item["vec"]))
            seen_litmus.add(litmus_name)
        self.logger.info(f"BO Batch Selected {len(best_batch)} unique candidates.")
        return best_batch

class LitmusRunnerForBayes(LitmusRunner):
    def __init__(
        self,
        litmus_list,
        param_space,
        stat_log,
        can_perple_path,
        perple_dict_path,
        standard_path,
        mode="time",
        init_samples_per_litmus=3,
        bo_iters=10000,
        pipeline_host="192.168.1.105",
        pipeline_user="root",
        pipeline_pass="riscv",
        pipeline_port=22
    ):
        super().__init__(litmus_list, [], stat_log, mode)
        self.can_perple_list = []
        standard_dict = load_litmus_dict(standard_path)
        with open(can_perple_path, "r") as f:
            can_perple_list = f.readlines()
            for can_perple in can_perple_list:
                self.can_perple_list.append(can_perple.strip())
        perple_files = get_files(perple_dict_path, "jsonl")
        self.perple_dict_list = {}
        for perple_file in perple_files:
            with open(perple_file, "r") as f:
                loaded_dict = json.load(f)
                self.perple_dict_list[perple_file.split("/")[-1][:-7]] = loaded_dict
        self.ps = param_space
        self.bo = RandomForestBO(param_space, litmus_list, litmus_vec_path, self.can_perple_list)

        self.init_samples_per_litmus = init_samples_per_litmus
        self.bo_iters = bo_iters

        self.total_runs = 0
        self.init_done = False

        self.cache = ResultCache(stat_log + ".cache.jsonl")
        self.error_cache = ErrorCache(stat_log + ".error.cache.jsonl")
        self.logger = get_logger(LOG_NAME)
        self.cold_init = True

        self.logger.info("Initializing Async Pipeline...")

        self.pipeline = LitmusPipeline(
            host=pipeline_host,
            port=pipeline_port,
            username=pipeline_user,
            password=pipeline_pass,
            remote_work_dir=remote_path # 确保板子上有这个目录
        )
        self.pipeline.start()

    def _submit_one(self, litmus, vec, perp_dict):
        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()
        # [关键] 把 vector 挂载到 params 上，方便结果回来时找回
        params._temp_vec = params.to_vector()

        litmus_file = f"{litmus_path}/{litmus}.litmus"


        self.pipeline.submit_task(
            litmus_path=litmus_file,
            params=params,
            litmus_dir_path=dir_path,  # 本地编译产物路径
            log_dir_path=dir_path,  # 本地结果日志路径
            run_time=100000,  # 运行次数/时间
            perp_dict=perp_dict
        )

    # === [新增] 辅助函数：解析 Log 算分 ===
    def _parse_log_to_score(self, log_path, litmus_name, has_perple, mode):
        """
        以前 get_score 里既跑又算分，现在 pipeline 跑完了，这里只负责算分。
        """
        try:
            # 这里的逻辑要从你原来的 get_score 或 read_log_to_summary 里提取
            # 示例：
            # res = read_log_to_summary(log_path)
            # return calculate_metric(res)

            # 模拟：假设 file size 代表分数 (请替换为你的真实逻辑)
            # with open(log_path, 'r') as f: ...
            if has_perple:
                return log_path, parse_log_by_mode_perple(log_path, mode)
            else:
                return log_path, parse_log_by_mode(log_path, mode)
        except Exception as e:
            self.logger.error(f"Parse error {log_path}: {e}")
            return -1

    def _preload_and_train(self):
        """
        直接从 self.cache.data 中提取数据进行预热，避免重复读取文件 IO。
        """
        self.logger.info("=== Preloading data from memory (self.cache) for Warm Start ===")

        loaded_count = 0

        # self.cache.data 的格式是: { "litmus_name|0.1,0.2,0.3": score, ... }
        # 我们需要遍历字典，把 key 解析回 litmus 和 param_vec

        for key, score in self.cache.data.items():
            try:
                # 1. 解析 Key
                # 对应 ResultCache._make_key: return f"{litmus}|" + ",".join(map(str, param_vec))
                if "|" not in key:
                    continue

                litmus, param_str = key.split("|", 1)

                # 2. 还原 Param Vector (字符串 -> float列表)
                if not param_str:
                    continue
                param_vec = [float(p) for p in param_str.split(",")]

                # 3. 加入 BO 训练集
                self.bo.add(litmus, param_vec, score)

                # 4. 更新计数 (用于控制采样平衡)
                self.bo.litmus_run_count[litmus] += 1

                loaded_count += 1
            except Exception as e:
                # 容错：防止因历史数据格式差异导致解析失败 crash
                # self.logger.warning(f"Failed to parse cache key: {key}, err: {e}")
                continue

        self.logger.info(f"Loaded {loaded_count} historical records from ResultCache.")

        if loaded_count > 0:
            self.logger.info(f"Training Initial Random Forest Model with {loaded_count} samples...")
            start_t = time.time()
            self.bo.fit()
            self.logger.info(f"Initial Training Complete. Time: {time.time() - start_t:.2f}s")
        else:
            self.logger.warning("Cache is empty in memory! BO will start from scratch (Cold Start).")



    def getNext(self):
        """
        BO-driven task selection
        Return: (litmus_name, params) or None
        """
        # assert False
        # ---------- 终止条件 ----------
        if self.total_runs >= self.bo_iters:
            return None

        # ---------- BO decision ----------
        choice = self.bo.select_batch_next()
        if choice is None:
            return None
        self.cold_init = False
        litmus, full_vec = choice
        param_dim = self.ps.dim
        param_vec = full_vec[:param_dim]

        return litmus, param_vec

        # === [重写] Run 方法 ===
    def run(self):
        self.logger.info("=== Starting Async BO Loop (Warm Start) ===")

        # 配置参数
        TRAIN_TRIGGER = 5  # 每收 5 个结果训练一次
        SUBMIT_SIZE = 8  # 每次生成 8 个新任务
        MAX_QUEUE_SIZE = 20  # 队列最大保持 20 个

        # --- 1. 初始填充 (Pre-fill) ---
        # 因为你有 _preload_and_train，模型已经训练过了 (Warm Start)
        # 所以我们可以直接让 BO 推荐任务，而不需要随机采样！
        self.logger.info("Pre-filling pipeline with BO suggestions...")

        initial_batch = self.bo.select_batch_next(batch_size=20)

        if not initial_batch:
            self.logger.warning("BO returned no suggestions! Fallback to random.")
            # 如果 BO 没吐出东西（极少见），手动随机填几个
            # ... (添加随机填充逻辑) ...
            assert False, "BO returned no suggestions!"
        else:
            for litmus, vec in initial_batch:
                self._submit_one(litmus, vec, self.perple_dict_list.get(litmus, None))

        # --- 2. 流式循环 ---
        pending_results_buffer = []

        # stream_results 是个生成器，会阻塞等待结果
        for result in self.pipeline.stream_results():
            if result is None:
                if self.total_runs >= self.bo_iters:
                    break
                if self.total_runs < self.bo_iters:
                    try:
                        # 1. 训练模型并生成新任务 (一次生成8个)
                        # 因为是 Warm Start，数据只会越来越多，BO 会越来越准
                        new_batch = self.bo.select_batch_next(batch_size=SUBMIT_SIZE)

                        # 2. 提交新任务
                        for (nxt_litmus, nxt_vec) in new_batch:
                            # 查重：如果 Cache 里已经有了，就没必要再跑了（虽然 select_batch 应该避免）
                            if self.cache.get(nxt_litmus, nxt_vec) is None:
                                self._submit_one(nxt_litmus, nxt_vec, self.perple_dict_list.get(nxt_litmus, None))

                        self.logger.info(f"Submitted {len(new_batch)} new tasks.")

                    except Exception as e:
                        self.logger.error(f"BO Step Failed: {e}")
                continue

            # === A. 处理结果 ===
            task = result['task']
            log_path = result['log_path']
            litmus_name = task.litmus_path.split("/")[-1][:-7]

            # 找回 vector
            if hasattr(task.params, '_temp_vec'):
                param_vec = task.params._temp_vec
            else:
                self.logger.error("Lost vector info in task!")
                continue

            # 算分
            _, score = self._parse_log_to_score(log_path, litmus_name, task.params.is_perple(), self.mode)
            self.logger.info(f"[FINISHED] {litmus_name} Score: {score:.4f}")

            # 存入数据 (Cache + BO)
            if score != -1:
                self.bo.add(litmus_name, param_vec, score)
                self.cache.add(litmus_name, param_vec, score)
                self.bo.litmus_run_count[litmus_name] += 1
                self.total_runs += 1

            # 加入缓冲区
            pending_results_buffer.append(score)

            # === B. 批次训练 & 更新 (满5个) ===
            if len(pending_results_buffer) >= TRAIN_TRIGGER:
                self.logger.info(f"Collected {len(pending_results_buffer)} results. Training...")

                if self.total_runs < self.bo_iters:
                    try:
                        # 1. 训练模型并生成新任务 (一次生成8个)
                        # 因为是 Warm Start，数据只会越来越多，BO 会越来越准
                        new_batch = self.bo.select_batch_next(batch_size=SUBMIT_SIZE)

                        # 2. 提交新任务
                        for (nxt_litmus, nxt_vec) in new_batch:
                            # 查重：如果 Cache 里已经有了，就没必要再跑了（虽然 select_batch 应该避免）
                            if self.cache.get(nxt_litmus, nxt_vec) is None:
                                self._submit_one(nxt_litmus, nxt_vec, self.perple_dict_list.get(nxt_litmus, None))

                        self.logger.info(f"Submitted {len(new_batch)} new tasks.")

                    except Exception as e:
                        self.logger.error(f"BO Step Failed: {e}")

                # 3. [关键] 优胜劣汰：清理 Compile Queue
                # 丢弃积压的旧任务，保证板子跑的都是最新的
                self.pipeline.keep_fresh(max_size=MAX_QUEUE_SIZE)

                # 4. 清空缓冲区
                pending_results_buffer.clear()

            # 退出条件
            if self.total_runs >= self.bo_iters:
                break

        self.logger.info("Run finished. Waiting for pipeline...")
        self.pipeline.wait_completion()
        return self.results




litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat_bayes.csv"
litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector_gt0.log"
can_perple_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/can_perple.log"
perple_dict_path = "/home/whq/Desktop/code_list/perple_test/perple_json"
standard_path = './log1_positive_scores.json'


# litmus_path = "/home/software/桌面/bayes/perple_test_riscv/all_allow_litmus_C910_naive"
# stat_log = "/home/software/桌面/bayes/perple_test_riscv/bayes_stat/log_record_bayes.log"
# dir_path = "/home/software/桌面/bayes/perple_test_riscv/bayes_log"
# log_path = "/home/software/桌面/bayes/perple_test_riscv/bayes_stat/log_stat_bayes.csv"
# can_perple_path = "./can_perple.log"
# litmus_vec_path="/home/software/桌面/bayes/perple_test_riscv/bayes_stat/litmus_vector_gt0.log"
# perple_dict_path = "./perple_json"
# standard_path = './log1_positive_scores.json'

host = "192.168.226.168"  # 远程服务器地址
# host = "10.42.0.131"
port = 22  # SSH 端口
username = "sipeed"  # SSH 用户名
password = "sipeed"  # SSH 密码
remote_path = "/home/sipeed/test"

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    ts = time.strftime("%Y%m%d-%H%M%S")

    logger = setup_logger(
        log_file=f"{stat_log}.{ts}.run.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"Start BO run | seed={SEED}")

    litmus_list = get_files(litmus_path)
    litmus_list = [path.split("/")[-1][:-7] for path in litmus_list]
    for litmus in litmus_list:
        print(litmus)
    param_space = LitmusParamSpace()
    runner = LitmusRunnerForBayes(
        litmus_list,
        param_space,
        stat_log,
        can_perple_path,
        perple_dict_path,
        standard_path= standard_path,
        pipeline_host=host,
        pipeline_user=username,
        pipeline_pass=password,
        pipeline_port=port
    )
    runner._preload_and_train()
    runner.run()
