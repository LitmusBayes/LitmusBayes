import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 核心模型定义 =================

class TransferRandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=200,
                 litmus_vec_path=["/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"]):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            min_samples_leaf=3,  # 保持一定的泛化性
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.is_new_data = []  # 核心：标记数据来源
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)

        # 加载 Litmus 特征向量
        self.litmus_to_vector_dict = {}
        for vec_path in litmus_vec_path:
            litmus_vec_dict = self.load_litmus_vectors(vec_path)
            for vec in litmus_vec_dict:
                self.litmus_to_vector_dict[vec] = litmus_vec_dict[vec]

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line: continue
                name, vec_str = line.split(":", 1)
                vec = eval(vec_str)
                litmus_to_vec[name] = list(vec)
        return litmus_to_vec

    def add(self, litmus_name, param_vec, score, is_new=False):
        """添加数据，并打上是否为新芯片数据的标签"""
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)
        self.is_new_data.append(is_new)

    def fit(self, new_data_weight_ratio=None):
        self.logger.info(f"Start fitting Transfer Model. Total samples: {len(self.X)}")

        # 转换到 Log 空间
        y_train_log = np.log1p(np.array(self.y))

        # ================= 样本加权核心逻辑 =================
        weights = np.ones(len(self.y))

        num_new = sum(1 for is_new in self.is_new_data if is_new)
        num_old = len(self.y) - num_new

        if num_new > 0 and num_old > 0:
            # 默认策略：让新数据的总权重等于旧数据的总权重
            if new_data_weight_ratio is None:
                weight_factor = num_old / num_new
            else:
                weight_factor = new_data_weight_ratio

            self.logger.info(f"Old data: {num_old}, New data: {num_new}")
            self.logger.info(f"Assigning weight factor {weight_factor:.2f} to new chip data.")

            for i, is_new in enumerate(self.is_new_data):
                if is_new:
                    weights[i] = weight_factor
        else:
            self.logger.info("Training with uniform weights (no mixed domain data detected).")

        # 带权重的模型拟合
        self.model.fit(np.array(self.X), y_train_log, sample_weight=weights)

    def predict_one(self, litmus_name, param_vec):
        if litmus_name not in self.litmus_to_vector_dict:
            return None
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)
        pred_log = self.model.predict([feature])[0]
        return np.expm1(pred_log)


# ================= 稳健参数选择器 =================

class RobustParamSelector:
    def __init__(self, model, param_space, default_param=None):
        self.model = model
        self.ps = param_space
        self.default_param = default_param if default_param else [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def get_forest_uncertainty(self, X):
        """获取所有决策树的预测，计算 Log 空间的均值和标准差"""
        per_tree_pred = [tree.predict(X) for tree in self.model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)
        means = np.mean(per_tree_pred, axis=0)
        stds = np.std(per_tree_pred, axis=0)
        return means, stds

    def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0):
        recommendations = {}
        all_param_vectors = np.array(self.ps.get_all_combinations())
        n_params = len(all_param_vectors)

        print(f"Searching best params from {n_params} combinations (Alpha={alpha})...")

        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            if litmus not in litmus_feature_map:
                continue

            l_feat = np.array(litmus_feature_map[litmus])
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])

            # 计算 LCB 稳健分数
            means_log, stds_log = self.get_forest_uncertainty(X_batch)
            robust_scores_log = means_log - (alpha * stds_log)

            best_idx = np.argmax(robust_scores_log)
            best_mean_log = means_log[best_idx]
            best_std_log = stds_log[best_idx]
            best_vec = all_param_vectors[best_idx]

            # 还原到真实分数
            predicted_real_score = np.expm1(best_mean_log)

            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = "optimized_transfer"
            else:
                final_vec = self.default_param
                decision_type = "default (safety fallback)"

            recommendations[litmus] = {
                "param": final_vec,
                "pred_score": float(predicted_real_score),
                "pred_std_log": float(best_std_log),
                "decision": decision_type
            }

        return recommendations


# ================= 主程序 =================

if __name__ == "__main__":
    # 配置路径
    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
    stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
    litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"

    # 你的数据文件
    old_cache_file = "../RQ1/cache.jsonl"
    new_chip_file = "log_record_init_banana_kmeans_norm.log.validation_cache.jsonl"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.transfer_eval.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Instance-Weighting Transfer Run | Seed={SEED} ===")

    # 1. 初始化模型
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    param_space = LitmusParamSpace()
    bo = TransferRandomForestBO(
        param_space,
        litmus_names,
        n_estimators=100,  # 维持100棵树即可，加速搜索
        litmus_vec_path=[litmus_vec_path]
    )

    # 2. 加载旧芯片数据 (50000条)
    logger.info(f"Loading OLD data from {old_cache_file} ...")
    old_data_count = 0
    if os.path.exists(old_cache_file):
        with open(old_cache_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        # 注意 is_new=False
                        bo.add(obj["litmus"], obj["param"], obj["score"], is_new=False)
                        old_data_count += 1
                    except:
                        pass
    logger.info(f"Loaded {old_data_count} old records.")

    # 3. 加载新芯片数据 (400条)
    logger.info(f"Loading NEW chip data from {new_chip_file} ...")
    new_data_count = 0
    if os.path.exists(new_chip_file):
        with open(new_chip_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        # 注意 is_new=True
                        bo.add(obj["litmus"], obj["param"], obj["score"], is_new=True)
                        new_data_count += 1
                    except:
                        pass
    logger.info(f"Loaded {new_data_count} new records.")

    # 4. 联合训练模型 (内部会自动计算高权重)
    t_start = time.time()
    bo.fit()
    logger.info(f"Transfer Model training finished in {time.time() - t_start:.2f}s")

    # 5. 执行稳健参数推荐
    logger.info("=" * 60)
    logger.info("Starting Robust Parameter Selection...")

    selector = RobustParamSelector(
        model=bo.model,
        param_space=param_space,
        default_param=[0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    )

    # 这里的 alpha=1.5 是一个比较稳妥的值，兼顾加速和防跌
    recommendations = selector.select_best_params(
        litmus_list=litmus_names,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=1.5
    )

    # 6. 统计与保存
    optimized_count = sum(1 for v in recommendations.values() if v['decision'] == 'optimized_transfer')
    default_count = len(recommendations) - optimized_count

    logger.info(f"Selection Finished.")
    logger.info(f"  - Optimized via Transfer: {optimized_count}")
    logger.info(f"  - Default (Safety Fallback): {default_count}")

    output_file = "best_params_recommendation_transfer.json"
    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_file}")
    logger.info("=== Done ===")