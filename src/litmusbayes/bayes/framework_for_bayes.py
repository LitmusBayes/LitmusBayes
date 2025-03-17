import json
import logging
import os
import random
import time
from collections import defaultdict

from scipy.stats import norm

from src.litmusbayes.bayes.framework import LitmusRunner, get_score
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.litmus_params import LitmusParams
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files, read_log_to_summary
import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np

SEED = 2025
LOG_NAME = "bayes"
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

    def __init__(self, param_space: LitmusParamSpace, litmus_list, can_perple_list, n_estimators=200,
                 litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"):
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
        self.max_runs_per_litmus = 10

        # ===== 关键：加载 litmus 向量 =====
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

        # ===== 一致性检查 =====
        for l in self.litmus_to_vector_dict:
            if l not in self.litmus_list:
                raise ValueError(f"Missing vector for litmus: {l}")

        self.logger = get_logger(LOG_NAME)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                name, vec_str = line.split(":", 1)
                vec = eval(vec_str)  # 你这个文件是可信的
                litmus_to_vec[name] = list(vec)

        return litmus_to_vec

    # 加入训练数据
    def add(self, litmus_name, param_vec, score):
        self.logger.info(f"Adding Train {litmus_name} with {param_vec} with score {score}")
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec)+list(litmus_vec))
        self.y.append(score)
        self.max_litmus_score[litmus_name] = max(self.max_litmus_score.get(litmus_name,0), score)

    # 训练 RF
    def fit(self):
        self.model.fit(np.array(self.X), np.array(self.y))

    # EI 计算
    def compute_ei(self, candidate_vecs, litmus_name):
        C = np.array(candidate_vecs)

        preds = np.array([
            est.predict(C) for est in self.model.estimators_
        ])

        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

        eta = self.max_litmus_score.get(litmus_name, -np.inf)

        ei = np.zeros_like(mu)

        # -------- sigma > 0：标准 EI --------
        mask = sigma > 1e-8
        if np.any(mask):
            z = (mu[mask] - eta) / sigma[mask]
            ei[mask] = (
                    (mu[mask] - eta) * norm.cdf(z)
                    + sigma[mask] * norm.pdf(z)
            )

        # -------- sigma == 0：退化为 exploitation --------
        ei[~mask] = np.maximum(mu[~mask] - eta, 0.0)
        # self.logger.info(f"Compute EI {litmus_name} get {ei}")
        return ei

    def compute_ucb(self, candidate_vecs, beta=1.5):
        C = np.array(candidate_vecs)

        preds = np.array([
            est.predict(C) for est in self.model.estimators_
        ])

        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

        return mu + beta * sigma

    def sample_litmus(self, n_litmus):
        """
        随机采样 litmus test（不超过最大运行次数）
        """
        # 1. 过滤掉已经跑满的 litmus
        available = [
            l for l in self.litmus_list
            if self.litmus_run_count[l] < self.max_runs_per_litmus
        ]

        if not available:
            return []

        # 2. 随机采样（不放回）
        k = min(n_litmus, len(available))
        sampled = random.sample(available, k)

        return sampled

    # 候选生成
    def generate_candidates(self, n_litmus=5, n_param=200):
        candidates = []

        litmus_names = self.sample_litmus(n_litmus)

        for litmus in litmus_names:
            litmus_vector = self.litmus_to_vector_dict[litmus]
            can_perple = True if litmus in self.can_perple_list else False
            for _ in range(n_param):
                p = self.ps.random_vector(can_perple)
                candidates.append((litmus, list(p)+list(litmus_vector)))


        return candidates


    def groupby_litmus(self, cands):
        """
        cands: List[(litmus_name, full_vec)]
        return: Dict[litmus_name, List[full_vec]]
        """
        groups = defaultdict(list)
        for litmus, vec in cands:
            groups[litmus].append(vec)
        return groups

    # 选 EI 最大的点作为下一次评估点
    def select_next(self):
        if len(self.X) == 0:
            return None
        self.fit()

        cands = self.generate_candidates()

        best = None
        best_ei = -np.inf

        for litmus, vecs in self.groupby_litmus(cands).items():

            X = vecs
            ei = self.compute_ei(X, litmus)
            idx = int(np.argmax(ei))

            if ei[idx] > best_ei:
                best_ei = ei[idx]
                best = (litmus, X[idx])

        self.logger.info(f"BO Select Next: {best}, ei: {best_ei}")
        return best

    def self_check(self, vec, score):
        X_sample = np.array(vec).reshape(1, -1)
        preds = np.array([
            est.predict(X_sample) for est in self.model.estimators_
        ])

        mu = preds.mean(axis=0)
        self.fit()
        preds = np.array([
            est.predict(X_sample) for est in self.model.estimators_
        ])

        mu_after = preds.mean(axis=0)
        self.logger.info(f"BO self check: {vec}, score: {score}, before: {mu}, after: {mu_after}")


class LitmusRunnerForBayes(LitmusRunner):
    def __init__(
        self,
        litmus_list,
        param_space,
        stat_log,
        can_perple_path,
        mode="time",
        init_samples_per_litmus=3,
        bo_iters=10000,
    ):
        super().__init__(litmus_list, [], stat_log, mode)

        self.can_perple_list = []
        with open(can_perple_path, "r") as f:
            can_perple_list = f.readlines()
            for can_perple in can_perple_list:
                self.can_perple_list.append(can_perple.strip())
        self.ps = param_space
        self.bo = RandomForestBO(param_space, litmus_list, self.can_perple_list)

        self.init_samples_per_litmus = init_samples_per_litmus
        self.bo_iters = bo_iters

        self.total_runs = 0
        self.init_done = False

        self.cache = ResultCache(stat_log + ".cache.jsonl")
        self.error_cache = ErrorCache(stat_log + ".error.cache.jsonl")
        self.logger = get_logger(LOG_NAME)
        self.cold_init = True



    def _initial_sample(self):
        for litmus in self.litmus_list:
            for _ in range(self.init_samples_per_litmus):
                can_perple = True if litmus in self.can_perple_list else False
                param_vec = self.ps.random_vector(can_perple= can_perple)
                # params = self.ps.vector_to_params(param_vec)

                yield litmus, param_vec

    def getNext(self):
        """
        BO-driven task selection
        Return: (litmus_name, params) or None
        """
        # assert False
        # ---------- 终止条件 ----------
        if self.total_runs >= self.bo_iters:
            return None

        # ---------- Cold start ----------
        if not self.init_done:
            if not hasattr(self, "_init_iter"):
                self._init_iter = iter(self._initial_sample())

            try:
                litmus, param_vec = next(self._init_iter)
                return litmus, param_vec
            except StopIteration:
                self.init_done = True
                # fall through to BO

        # ---------- BO decision ----------
        choice = self.bo.select_next()
        if choice is None:
            return None
        self.cold_init = False
        litmus, full_vec = choice
        param_dim = self.ps.dim
        param_vec = full_vec[:param_dim]

        return litmus, param_vec

    def run(self):
        """
        BO-driven execution loop
        """
        while True:
            nxt = self.getNext()
            if nxt is None:
                break

            litmus, param_vec = nxt
            params = self.ps.vector_to_params(param_vec)
            params.set_riscv_gcc()
            print(f"[RUN] litmus={litmus}, params={params.to_dict()}")
            if self.error_cache.has(litmus, param_vec):
                self.logger.info(
                    f"[SKIP-ERROR] litmus={litmus}, params={param_vec}"
                )
                continue
            # ---------- 查 cache ----------
            cached = self.cache.get(litmus, param_vec)
            if cached is not None:
                score = cached
                print(f"[CACHE HIT] litmus={litmus}, score={score:.4f}")
            else:
                # assert False
                print(f"[CHIP RUN] litmus={litmus}, params={params.to_dict()}")
                log_path, score = get_score(f"{litmus_path}/{litmus}.litmus", params, mode=self.mode)
                if log_path is None:
                    self.logger.warning(
                        f"[ERROR] litmus={litmus}, params={param_vec}, err=error"
                    )
                    self.error_cache.add(
                        litmus,
                        param_vec,
                        error_type="runtime_error",
                        error_msg="error",
                    )
                    continue
                self.cache.add(litmus, param_vec, score)

            self.logger.info(f"Score {litmus} is: {score}")

            # 回传给 BO
            self.bo.add(litmus, param_vec, score)
            self.bo.litmus_run_count[litmus] += 1

            self.total_runs += 1
            self.results.append((litmus, params, score))

            print(
                f"[DONE] score={score:.4f}, "
                f"best[{litmus}]={self.bo.max_litmus_score[litmus]:.4f}"
            )
            if not self.cold_init:
                self.bo.self_check(param_vec+self.bo.litmus_to_vector_dict[litmus],score)
        return self.results




litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat_bayes.csv"
can_perple_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/can_perple.log"
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
    runner = LitmusRunnerForBayes(litmus_list, param_space, stat_log, can_perple_path)
    runner.run()
