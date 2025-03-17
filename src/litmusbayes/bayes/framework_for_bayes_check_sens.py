import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


class SensitivityAwareRFBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=500,
                 litmus_vec_path=None):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            min_samples_leaf=2,
            random_state=SEED,
        )

        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)
        self.scaler = StandardScaler()

        # 存储每个 Litmus Test 的敏感度指纹
        self.sensitivity_map = {}

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if not os.path.exists(path): return {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line: continue
                name, vec_str = line.split(":", 1)
                try:
                    vec = eval(vec_str)
                    litmus_to_vec[name] = list(vec)
                except:
                    pass
        return litmus_to_vec

    # === 关键步骤 1: 从全量数据中挖掘参数敏感度 ===
    # 这就是你想要的：算出哪个参数一变动，性能就下降
    def precompute_sensitivity(self, all_data):
        self.logger.info("Computing Sensitivity Fingerprints from 60k data...")

        # 1. 按 Litmus 分组
        groups = defaultdict(list)
        for item in all_data:
            groups[item['litmus']].append(item)

        count = 0
        for name, records in groups.items():
            if len(records) < 5:
                # 数据太少算不出相关性，给个全0向量
                # 假设 Param 维度是 11 (根据你的 Param Space)
                self.sensitivity_map[name] = [0.0] * 11
                continue

            # 提取 Param 矩阵 (N, 11) 和 Score (N,)
            params = np.array([r['param'] for r in records])
            scores = np.array([r['score'] for r in records])

            # 如果分数完全不动（死水），那敏感度全是0
            if np.std(scores) < 1e-6:
                self.sensitivity_map[name] = [0.0] * params.shape[1]
                continue

            sens_vec = []
            # 对每一个参数维度，计算它和 Score 的相关性
            for col_idx in range(params.shape[1]):
                col_values = params[:, col_idx]

                # 如果这个参数在所有样本里都没变过，相关性为0
                if np.std(col_values) < 1e-6:
                    sens_vec.append(0.0)
                else:
                    # 使用 Spearman 相关系数 (绝对值)
                    # 绝对值越大 -> 稍微一动，分数就变 -> 敏感度越高！
                    corr, _ = spearmanr(col_values, scores)
                    if np.isnan(corr): corr = 0.0
                    sens_vec.append(abs(corr))

            self.sensitivity_map[name] = sens_vec
            count += 1

        self.logger.info(f"Computed sensitivity for {count} tests.")

    # === 关键步骤 2: 将敏感度注入特征 ===
    def _compute_features(self, param_vec, litmus_vec, litmus_name):
        p = np.array(param_vec)
        l = np.array(litmus_vec)

        # 获取该 Test 的敏感度指纹 (11维)
        # 例如: [0.9, 0.0, 0.1 ...] 表示第0个参数极度敏感
        sens = np.array(self.sensitivity_map.get(litmus_name, [0.0] * len(p)))

        # 1. 基础特征
        base_features = list(p) + list(l) + list(sens)

        # 2. **加权特征 (Weighted Param)** —— 这就是你在找的“权重”！
        # 逻辑：我们将 参数值 * 敏感度
        # 如果敏感度是 0，这个特征就变成了 0 (模型会忽略它)
        # 如果敏感度是 1，这个特征就被高亮保留 (模型会重点关注它)
        weighted_param = (p * sens).tolist()

        # 3. 显式交叉 (Litmus * Param)
        interaction = np.outer(p, l).flatten().tolist()

        return base_features + weighted_param + interaction

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict: return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]

        features = self._compute_features(param_vec, litmus_vec, litmus_name)
        self.X.append(features)
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting Sensitivity RF (Features: {len(self.X[0])})...")
        X_np = np.array(self.X)
        y_train = np.log1p(np.array(self.y))
        self.X_scaled = self.scaler.fit_transform(X_np)
        self.model.fit(self.X_scaled, y_train)

    def predict_batch(self, litmus_list, param_list):
        X_batch = []
        valid_indices = []
        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]
                features = self._compute_features(param, litmus_vec, litmus)
                X_batch.append(features)
                valid_indices.append(i)

        if not X_batch: return [], []
        X_batch_scaled = self.scaler.transform(np.array(X_batch))
        pred_log = self.model.predict(X_batch_scaled)
        return np.expm1(pred_log), valid_indices


# ================= 辅助分析函数 =================

def analyze_ranking_quality(groups_data, output_dir="./analysis_plots"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    regret_list = []
    for litmus, records in groups_data.items():
        if len(records) < 2: continue
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        gt_best_val = records_sorted_by_actual[0]['actual']
        if gt_best_val <= 0: continue
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        model_pick_val = records_sorted_by_pred[0]['actual']
        regret = (gt_best_val - model_pick_val) / gt_best_val
        regret_list.append(regret)
    print("\n" + "=" * 40)
    print("      SENSITIVITY MODEL DIAGNOSIS      ")
    print("=" * 40)
    regret_arr = np.array(regret_list)
    print(f"Mean Performance Regret: {np.mean(regret_arr) * 100:.2f}%")
    print(f"Median Performance Regret: {np.median(regret_arr) * 100:.2f}%")
    print(f"Top-1 Hit Rate (Zero Regret): {np.mean(regret_arr == 0) * 100:.2f}%")


def analyze_topk_recall(groups_data):
    k_values = [1, 3, 5, 10]
    hits = {k: 0 for k in k_values}
    total_groups = 0
    for litmus, records in groups_data.items():
        if len(records) < 2: continue
        total_groups += 1
        max_actual = max(r['actual'] for r in records)
        sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        for k in k_values:
            candidates = sorted_by_pred[:k]
            if any(c['actual'] >= max_actual for c in candidates):
                hits[k] += 1
    print("\n" + "=" * 40)
    print("      TOP-K RECALL ANALYSIS      ")
    print("=" * 40)
    for k in k_values:
        print(f"Top-{k:<2} Recall: {hits[k] / total_groups * 100:.2f}%")


if __name__ == "__main__":
    # Setup
    random.seed(SEED)
    np.random.seed(SEED)
    stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
    litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
    cache_file_path = stat_log_base + ".cache4_norm.jsonl"

    log_file_name = f"{stat_log_base}.sensitivity_rf.run.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Evaluation Run (Sensitivity Aware) | Seed={SEED} ===")

    # 读取数据
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]
    param_space = LitmusParamSpace()

    # 3. 初始化新模型
    bo = SensitivityAwareRFBO(
        param_space,
        litmus_names,
        n_estimators=500,
        litmus_vec_path=litmus_vec_path
    )

    # 4. 加载全量数据
    logger.info(f"Loading data from {cache_file_path} ...")
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass
    else:
        exit(1)

    # === 核心步骤：先用全量数据计算敏感度指纹 ===
    # 这就是你说的 "先用这6w条数据知道哪些维度很重要"
    bo.precompute_sensitivity(all_data)

    # 5. 切分数据
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    # 6. 训练
    logger.info("Building training set...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])
    bo.fit()

    # 7. 预测
    logger.info("Evaluating on test set...")
    test_litmus_names = [item["litmus"] for item in test_data]
    test_params = [item["param"] for item in test_data]

    t0 = time.time()
    preds, indices_keep = bo.predict_batch(test_litmus_names, test_params)
    logger.info(f"Prediction finished. Valid samples: {len(preds)}")

    # 8. 映射结果
    groups = defaultdict(list)
    for ptr, original_idx in enumerate(indices_keep):
        item = test_data[original_idx]
        record = {'param': item["param"], 'actual': item["score"], 'pred': preds[ptr]}
        groups[item["litmus"]].append(record)

    # 9. 结果分析
    analyze_ranking_quality(groups)
    analyze_topk_recall(groups)