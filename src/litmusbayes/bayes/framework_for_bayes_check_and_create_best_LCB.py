import json
import logging
import os
import random
import time
from collections import defaultdict
from scipy.stats import spearmanr, norm

# 引入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np

SEED = 2025
LOG_NAME = "bayes_eval_norm"

# ================= 关键参数配置 =================
# 风险惩罚系数 (Alpha)
# 归一化后，Sigma 通常在 0.1~0.3 之间。
# Alpha = 1.0 ~ 2.0 是比较合理的范围。
ALPHA = 2.0


# ================= 类定义 =================

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
        pass


class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=100,
                 litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"):
        self.ps = param_space
        # 注意：为了计算不确定性，我们需要稍微控制树的深度，防止过拟合导致方差计算失真
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            max_features="sqrt",
            max_depth=12,  # 限制深度，避免死记硬背
            min_samples_leaf=5,  # 增加叶子节点样本数，提高泛化能力
            random_state=SEED
        )

        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)

        # 加载向量
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

        # === 新增：用于归一化的极值记录 ===
        self.y_min = 0.0
        self.y_max = 1.0

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line: continue
                    try:
                        name, vec_str = line.split(":", 1)
                        vec = eval(vec_str)
                        litmus_to_vec[name] = list(vec)
                    except:
                        pass
        return litmus_to_vec

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting with {len(self.X)} samples...")

        X_arr = np.array(self.X)
        y_arr = np.array(self.y)

        # === 修改：使用 MinMax 归一化替代 Log ===
        self.y_min = y_arr.min()
        self.y_max = y_arr.max()

        # 防止 max == min 导致的除以零
        if self.y_max == self.y_min:
            self.y_max += 1e-6

        # 将 y 映射到 [0, 1] 区间
        y_norm = (y_arr - self.y_min) / (self.y_max - self.y_min)

        self.logger.info(f"Data Normalized: Min={self.y_min:.4f}, Max={self.y_max:.4f}")

        # 训练模型
        self.model.fit(X_arr, y_norm)

    def predict_one(self, litmus_name, param_vec):
        """用于评估阶段的简单预测 (只看均值)"""
        if litmus_name not in self.litmus_to_vector_dict:
            return None
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)

        # 预测得到的是 0~1 的值
        pred_norm = self.model.predict([feature])[0]

        # 反归一化回真实尺度
        return pred_norm * (self.y_max - self.y_min) + self.y_min

    # ============ 修改：基于归一化的鲁棒批量预测 ============
    def predict_batch_robust(self, litmus_name, candidate_params_matrix):
        """
        输入:
            litmus_name: 测例名
            candidate_params_matrix: numpy array (N_candidates, param_dim)
        输出:
            best_idx: 最优参数在 candidates 中的索引
            best_score_pred: 预测的均值分数 (还原回真实尺度)
            uncertainty: 该点的归一化标准差 (0~1之间)
        """
        if litmus_name not in self.litmus_to_vector_dict:
            return None, 0, 0

        litmus_vec = self.litmus_to_vector_dict[litmus_name]

        # 1. 构造特征矩阵 X_batch
        X_litmus_part = np.tile(litmus_vec, (len(candidate_params_matrix), 1))
        X_batch = np.hstack([candidate_params_matrix, X_litmus_part])

        # 2. 获取森林中每棵树的预测值
        # 注意：现在的预测值本身就在 [0, 1] 之间
        all_preds = []
        for tree in self.model.estimators_:
            preds = tree.predict(X_batch)
            all_preds.append(preds)

        all_preds = np.array(all_preds)

        # 3. 计算均值和标准差 (在 [0, 1] 归一化空间内)
        mu_norm = np.mean(all_preds, axis=0)
        sigma_norm = np.std(all_preds, axis=0)

        # 4. 计算 LCB (Lower Confidence Bound) 分数
        # 公式：Score = Mean - (Alpha * Std)
        # 解释：在 0~1 的空间里，如果 Std 很大(比如0.3)，得分会被大幅扣减
        lcb_score_norm = mu_norm - (ALPHA * sigma_norm)

        # 5. 找到 LCB 最高的索引
        best_idx = np.argmax(lcb_score_norm)

        # 6. 准备返回结果
        best_mu_norm = mu_norm[best_idx]
        best_sigma_norm = sigma_norm[best_idx]

        # === 关键：反归一化 ===
        # 我们返回"预测均值"给用户看，而不是"LCB得分"，因为用户想看真实分数的预期
        best_score_real = best_mu_norm * (self.y_max - self.y_min) + self.y_min

        return best_idx, best_score_real, best_sigma_norm


# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_dnn_gt0.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"

if __name__ == "__main__":
    # 1. Setup Logger
    random.seed(SEED)
    np.random.seed(SEED)

    log_file_name = f"{stat_log_base}.check_norm.run.log"
    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Evaluation Run (Normalized LCB) | Seed={SEED} | Alpha={ALPHA} ===")

    # 2. 读取 Litmus List
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=100,  # 评估时可以用稍微少一点的树加快速度，或者保持200
        litmus_vec_path=litmus_vec_path
    )

    # 4. 加载 Cache
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
        logger.error("Cache file not found!")
        exit(1)

    logger.info(f"Total records loaded: {len(all_data)}")

    # 5. 切分数据 (这里为了生成最佳参数，我们使用全量数据进行训练会更好)
    # 但为了保持代码结构一致，我们直接全部加进去

    # 6. 构建训练集 & 训练
    logger.info("Building training set (Using ALL data for optimization)...")
    for item in all_data:
        bo.add(item["litmus"], item["param"], item["score"])

    # 此时模型训练数据会被归一化到 0~1
    bo.fit()

    logger.info("=" * 60)
    logger.info("STARTING ROBUST OPTIMIZATION (RISK-AWARE SEARCH)")
    logger.info(f"Policy: Maximize (Mean_Norm - {ALPHA} * Std_Norm)")
    logger.info("=" * 60)

    # 1. 生成全量参数候选集
    logger.info("Generating all possible parameter combinations...")
    all_candidates = param_space.get_all_combinations()
    logger.info(f"Total candidate combinations: {len(all_candidates)}")

    # 转为 numpy 矩阵
    X_base_params = np.array(all_candidates)

    # 2. 准备结果容器
    optimization_results = {}

    count = 0
    total_tasks = len(litmus_names)

    for litmus in litmus_names:
        count += 1
        if count % 100 == 0:
            logger.info(f"Processing {count}/{total_tasks}...")

        # 调用新的 Robust 预测方法
        best_idx, best_score_mean, best_score_std = bo.predict_batch_robust(litmus, X_base_params)

        if best_idx is None:
            continue

        best_param_vec = all_candidates[best_idx]

        # 记录结果
        optimization_results[litmus] = {
            "param": best_param_vec,
            "pred_score": float(best_score_mean),
            "uncertainty_norm": float(best_score_std)  # 这个值现在是 0~1 之间的标准差
        }

    # 4. 输出保存
    logger.info("=" * 60)
    logger.info(f"Optimization Completed for {len(optimization_results)} files.")

    output_file = "best_params_recommendation_norm.json"
    with open(output_file, "w") as f:
        json.dump(optimization_results, f, indent=4)

    logger.info(f"Results saved to {output_file}")
    logger.info("You can now run the validation script using this JSON.")

    # 打印几个示例
    logger.info("Sample Recommendations:")
    for k in list(optimization_results.keys())[:5]:
        res = optimization_results[k]
        logger.info(f"File: {k}")
        logger.info(f"  -> Param: {res['param']}")
        logger.info(f"  -> Exp Score (Real Scale): {res['pred_score']:.4f}")
        logger.info(f"  -> Uncertainty (0-1 Scale): {res['uncertainty_norm']:.4f}")