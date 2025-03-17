import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

# 引入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

# 假设这些是你原本的项目引用
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= [必须保留] 距离安全卫士 =================
# 这是防止在没见过的 30% 测试上跑出慢10倍结果的关键
class DistanceGuard:
    def __init__(self, train_vectors):
        self.train_vectors = np.array(train_vectors)
        if len(self.train_vectors) == 0:
            # 这是一个容错处理，防止训练集为空
            self.train_vectors = np.zeros((1, 1))
            self.avg_train_dist = 999.9
            return

        # 使用 KNN 建立索引
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(self.train_vectors)

        # 计算训练集内部的平均距离基准
        # 只有当样本数 > 1 时才能计算内部距离
        if len(self.train_vectors) > 1:
            nbrs_check = NearestNeighbors(n_neighbors=min(2, len(self.train_vectors)), algorithm='auto').fit(
                self.train_vectors)
            dists, _ = nbrs_check.kneighbors(self.train_vectors)
            # dists[:, 1] 是到最近邻居的距离
            self.avg_train_dist = np.mean(dists[:, -1])
        else:
            self.avg_train_dist = 0.0

        print(f"[DistanceGuard] Average inner-distance in training set: {self.avg_train_dist:.4f}")

    def is_safe(self, target_vector, threshold_factor=2.0):
        if self.avg_train_dist == 999.9: return False, 999.9  # 未初始化

        target_vector = np.array(target_vector).reshape(1, -1)
        distances, _ = self.nbrs.kneighbors(target_vector)
        dist = distances[0][0]

        limit = self.avg_train_dist * threshold_factor
        return dist <= limit, dist


# ================= RandomForestBO (保持不变) =================

class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, n_estimators=100,
                 litmus_vec_path=[]):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            min_samples_leaf=3,
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.train_vectors = []

        self.logger = get_logger(LOG_NAME)

        # 加载向量
        self.litmus_to_vector_dict = {}
        for vec_path in litmus_vec_path:
            if isinstance(vec_path, list):
                paths = vec_path
            else:
                paths = [vec_path]

            litmus_vec_dict = self.load_litmus_vectors(paths if isinstance(paths, str) else vec_path)
            self.litmus_to_vector_dict.update(litmus_vec_dict)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if os.path.exists(path):
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

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)
        self.train_vectors.append(list(litmus_vec))

    def fit(self):
        self.logger.info(f"Start fitting RF with {len(self.X)} samples...")
        if len(self.X) == 0:
            self.logger.warning("No data to fit!")
            return
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    def get_unique_train_vectors(self):
        if not self.train_vectors: return []
        unique_vecs = set(tuple(v) for v in self.train_vectors)
        return [list(v) for v in unique_vecs]


# ================= RobustParamSelector (逻辑保持不变) =================

class RobustParamSelector:
    def __init__(self, model, param_space, safety_guard=None, default_param=None):
        self.model = model
        self.ps = param_space
        self.safety_guard = safety_guard
        self.default_param = default_param if default_param else [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def get_forest_uncertainty(self, X):
        batch_size = 5000
        n_samples = X.shape[0]
        means_list = []
        stds_list = []

        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i + batch_size]
            per_tree_pred = [tree.predict(X_batch) for tree in self.model.estimators_]
            per_tree_pred = np.stack(per_tree_pred)
            means_list.append(np.mean(per_tree_pred, axis=0))
            stds_list.append(np.std(per_tree_pred, axis=0))

        return np.concatenate(means_list), np.concatenate(stds_list)

    def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0, dist_threshold=2.0):
        recommendations = {}
        all_param_vectors = np.array(self.ps.get_all_combinations())
        n_params = len(all_param_vectors)

        print(f"Selecting params for {len(litmus_list)} UNSEEN tests...")

        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            if litmus not in litmus_feature_map:
                recommendations[litmus] = {"param": self.default_param, "decision": "missing_feature"}
                continue

            l_feat = np.array(litmus_feature_map[litmus])

            # 1. 距离卫士拦截
            if self.safety_guard:
                is_safe, dist = self.safety_guard.is_safe(l_feat, threshold_factor=dist_threshold)
                if not is_safe:
                    recommendations[litmus] = {
                        "param": self.default_param,
                        "pred_score": -1,
                        "decision": f"OOD_fallback (dist={dist:.2f})"
                    }
                    continue

            # 2. 预测
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])
            means_log, stds_log = self.get_forest_uncertainty(X_batch)

            # 3. LCB 选择
            robust_scores_log = means_log - (alpha * stds_log)
            best_idx = np.argmax(robust_scores_log)

            best_mean_log = means_log[best_idx]
            predicted_real_score = np.expm1(best_mean_log)

            # 4. 收益阈值判断 (必须有明显收益才冒险)
            if predicted_real_score > 1.05:
                recommendations[litmus] = {
                    "param": all_param_vectors[best_idx].tolist(),
                    "pred_score": float(predicted_real_score),
                    "decision": "optimized"
                }
            else:
                recommendations[litmus] = {
                    "param": self.default_param,
                    "pred_score": float(predicted_real_score),
                    "decision": "low_gain_fallback"
                }

        return recommendations

# 配置路径

litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"

stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"

litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"

litmus_vec_path_new = "./make_new_litmus/litmus_vector.log"

litmus_path_new = "./make_new_litmus/litmus_output"

cache_file_path = stat_log_base + ".cache4_norm.jsonl"
# ================= 重写的 Main 函数 (按 Test 切分) =================

if __name__ == "__main__":
    # 1. 基础设置
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.zero_shot_eval.log"

    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Zero-Shot Evaluation (70% Train / 30% Test) | Seed={SEED} ===")

    # 2. 初始化 BO 和 参数空间
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        n_estimators=100,
        litmus_vec_path=[litmus_vec_path, litmus_vec_path_new]
    )

    # 3. 加载所有数据记录
    logger.info(f"Loading raw data from {cache_file_path} ...")
    all_data_records = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        all_data_records.append(obj)
                    except:
                        pass
    else:
        logger.error("Cache file not found!")
        exit(1)

    # 4. === 核心修改：按 Litmus Name 进行切分 ===

    # 提取所有出现过的 unique litmus names
    all_unique_litmus = list(set(d['litmus'] for d in all_data_records))
    all_unique_litmus.sort()  # 先排序保证确定性
    random.shuffle(all_unique_litmus)  # 再打乱

    total_litmus_count = len(all_unique_litmus)
    split_idx = int(total_litmus_count * 0.7)

    train_litmus_names = set(all_unique_litmus[:split_idx])
    test_litmus_names = list(all_unique_litmus[split_idx:])  # 这是我们要预测的那 30%

    logger.info(f"Total Unique Litmus Tests: {total_litmus_count}")
    logger.info(f"Training on: {len(train_litmus_names)} tests (70%)")
    logger.info(f"Holding out: {len(test_litmus_names)} tests (30%) for final validation")

    # 5. 构建训练数据集
    train_records = []
    for record in all_data_records:
        if record['litmus'] in train_litmus_names:
            train_records.append(record)

    logger.info(f"Constructed Train Dataset: {len(train_records)} records.")

    # 6. 训练模型
    for item in train_records:
        bo.add(item["litmus"], item["param"], item["score"])

    t_start = time.time()
    bo.fit()
    logger.info(f"Model training finished in {time.time() - t_start:.2f}s")

    # 7. 初始化安全卫士 (基于 70% 的训练集特征)
    logger.info("Initializing Distance Safety Guard...")
    unique_train_vecs = bo.get_unique_train_vectors()
    safety_guard = DistanceGuard(unique_train_vecs)

    # 8. === 只针对那 30% Unseen Tests 进行参数推荐 ===
    logger.info("=" * 60)
    logger.info("Running Optimization for HELD-OUT (Unseen) tests only...")

    selector = RobustParamSelector(
        bo.model,
        param_space,
        safety_guard=safety_guard,
        default_param=[0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    )

    # 这里 litmus_list 传入的是 test_litmus_names (那 30%)
    recommendations = selector.select_best_params(
        litmus_list=test_litmus_names,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=2.0,  # 保持保守
        dist_threshold=2.0  # 保持距离检测
    )

    # 9. 统计与保存
    optimized_count = sum(1 for v in recommendations.values() if v['decision'] == 'optimized')
    default_count = len(recommendations) - optimized_count

    logger.info(f"Zero-Shot Prediction Finished.")
    logger.info(f"  - Optimized: {optimized_count}")
    logger.info(f"  - Default/Fallback: {default_count}")

    output_file = "./make_new_litmus/best_params_unseen_test_set.json"
    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations for UNSEEN tests saved to: {output_file}")