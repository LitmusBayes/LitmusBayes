import json
import logging
import os
import random
import time
from collections import defaultdict

from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
from scipy.stats import norm, spearmanr

# 引入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import json
from tqdm import tqdm
SEED = 2025
LOG_NAME = "bayes_eval"


class DistanceGuard:
    """
    专门负责计算 '新测试用例' 与 '训练集' 之间的相似度。
    如果距离太远，说明模型是在 '瞎猜'，必须强行拦截。
    """

    def __init__(self, train_vectors):
        """
        :param train_vectors: 训练集中所有 Litmus Test 的特征向量 (List of lists)
        """
        self.train_vectors = np.array(train_vectors)
        if len(self.train_vectors) == 0:
            raise ValueError("Training vectors cannot be empty!")

        # 使用 KNN 建立索引，计算欧氏距离
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(self.train_vectors)

        # 计算训练集内部的平均距离，作为阈值的参考基准
        distances, _ = self.nbrs.kneighbors(self.train_vectors)
        # distances[:, 1] 是每个点到它最近邻居的距离 (排除自己)
        # 注意：如果 n_neighbors=1，kneighbors 返回的是自己到自己的距离(0)。
        # 这里为了计算阈值，我们需要找 n_neighbors=2
        nbrs_check = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(self.train_vectors)
        dists, _ = nbrs_check.kneighbors(self.train_vectors)
        self.avg_train_dist = np.mean(dists[:, 1])
        print(f"[DistanceGuard] Average inner-distance in training set: {self.avg_train_dist:.4f}")

    def is_safe(self, target_vector, threshold_factor=2.0):
        """
        判断目标向量是否足够 '熟悉'
        :param threshold_factor: 允许距离是训练集平均距离的多少倍。建议 1.5 ~ 3.0
        """
        target_vector = np.array(target_vector).reshape(1, -1)
        distances, _ = self.nbrs.kneighbors(target_vector)
        dist = distances[0][0]

        limit = self.avg_train_dist * threshold_factor

        # 返回是否安全，以及实际距离
        return dist <= limit, dist

# ================= 类定义 =================

class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    key = self._make_key(obj["litmus"], obj["param"])
                    self.data[key] = obj["score"]
        self.f = open(path, "a")

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        pass


class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=200,
                 litmus_vec_path=[]):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            min_samples_leaf=3,  # 增加叶子节点最小样本数，防止过拟合
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.train_vectors = []  # [新增] 专门存储训练所用的 litmus vector

        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)

        # 加载向量
        self.litmus_to_vector_dict = {}
        for vec_path in litmus_vec_path:
            # 兼容单个路径字符串或列表
            if isinstance(vec_path, list):
                paths = vec_path
            else:
                paths = [vec_path]

            # 这里原本的代码逻辑有一点小问题，我修正了一下循环
            litmus_vec_dict = self.load_litmus_vectors(paths if isinstance(paths, str) else vec_path)
            self.litmus_to_vector_dict.update(litmus_vec_dict)

    def load_litmus_vectors(self, path):
        # 你的原始代码似乎有点乱，这里简化一下逻辑
        litmus_to_vec = {}
        # 如果 path 是 list 里的一个元素(str)
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

        # [新增] 记录这仅仅是 Litmus 的特征向量，用于 DistanceGuard
        # 注意：这里会重复存很多次相同的 litmus vector，但这不影响 KNN 训练，或者你可以去重
        self.train_vectors.append(list(litmus_vec))

    def fit(self):
        self.logger.info(f"Start fitting RF with {len(self.X)} samples...")
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    # [新增] 获取去重后的训练向量，用于给 Guard 训练
    def get_unique_train_vectors(self):
        # 将 list 转 tuple 以便 set 去重，再转回 list
        unique_vecs = set(tuple(v) for v in self.train_vectors)
        return [list(v) for v in unique_vecs]
# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
litmus_vec_path_new = "./make_new_litmus/litmus_vector.log"
litmus_path_new = "./make_new_litmus/litmus_output"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"


# ================= 增强版 RobustParamSelector =================

class RobustParamSelector:
    def __init__(self, model, param_space, safety_guard=None, default_param=None):
        self.model = model
        self.ps = param_space
        self.safety_guard = safety_guard  # [新增] 注入 Guard
        self.default_param = default_param if default_param else [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def get_forest_uncertainty(self, X):
        # 优化：设置 batch size 以防止内存溢出
        batch_size = 5000
        n_samples = X.shape[0]
        means_list = []
        stds_list = []

        # 分批处理
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i + batch_size]
            per_tree_pred = [tree.predict(X_batch) for tree in self.model.estimators_]
            per_tree_pred = np.stack(per_tree_pred)
            means_list.append(np.mean(per_tree_pred, axis=0))
            stds_list.append(np.std(per_tree_pred, axis=0))

        return np.concatenate(means_list), np.concatenate(stds_list)

    def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0, dist_threshold=2.0):
        recommendations = {}

        # 预计算所有可能的参数组合 (约 2w 个)
        all_param_vectors = np.array(self.ps.get_all_combinations())
        n_params = len(all_param_vectors)

        logger.info(f"Selection Strategy: LCB(alpha={alpha}) + DistanceGuard(thresh={dist_threshold})")

        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            # 1. 基础检查
            if litmus not in litmus_feature_map:
                recommendations[litmus] = {"param": self.default_param, "decision": "missing_feature"}
                continue

            l_feat = np.array(litmus_feature_map[litmus])

            # ============ [新增] 第一道防线：距离检测 ============
            if self.safety_guard:
                is_safe, dist = self.safety_guard.is_safe(l_feat, threshold_factor=dist_threshold)
                if not is_safe:
                    # 距离太远，直接使用默认参数，不再浪费时间做预测
                    recommendations[litmus] = {
                        "param": self.default_param,
                        "pred_score": -1,
                        "decision": f"OOD_fallback (dist={dist:.2f})"  # Out Of Distribution
                    }
                    continue
            # =================================================

            # 2. 构造输入 (Params + Feat)
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])

            # 3. 预测 (Mean & Std)
            means_log, stds_log = self.get_forest_uncertainty(X_batch)

            # 4. LCB 计算
            robust_scores_log = means_log - (alpha * stds_log)
            best_idx = np.argmax(robust_scores_log)

            best_mean_log = means_log[best_idx]
            predicted_real_score = np.expm1(best_mean_log)

            # 5. 阈值判断 (第二道防线)
            if predicted_real_score > 1.05:  # 稍微提高一点门槛，比如要有 5% 的提升才采纳
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

# ================= 重写的 Main 函数 =================

if __name__ == "__main__":
    # 1. 基础设置
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)  # 如果用到了torch

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.robust_eval.log"

    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Robust Evaluation Run | Seed={SEED} ===")

    # 2. 读取 Litmus 文件列表
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    # 提取文件名作为 ID
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]
    logger.info(f"Found {len(litmus_names)} litmus files.")

    new_litmus_list = get_files(litmus_path_new, '.litmus')
    litmus_names_new = [path.split("/")[-1][:-7] for path in new_litmus_list]

    # 3. 初始化 BO 和 参数空间
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=100,  # 稍微降低一点树的数量以加快 Robust 搜索速度，或者保持 200
        litmus_vec_path=[litmus_vec_path, litmus_vec_path_new]
    )

    # 4. 加载训练数据 (Cache)
    logger.info(f"Loading training data from {cache_file_path} ...")
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        all_data.append(obj)
                    except:
                        pass
    else:
        logger.error("Cache file not found! Cannot train model.")
        exit(1)

    logger.info(f"Total records loaded: {len(all_data)}")

    # 打乱并切分数据 (保留一部分用于评估模型精度)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)  # 80% 训练
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    logger.info(f"Train size: {len(train_data)} | Test size: {len(test_data)}")

    # 5. 训练模型
    logger.info("Building dataset and fitting model...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])

    t_start = time.time()
    bo.fit()
    logger.info(f"Model training finished in {time.time() - t_start:.2f}s")

    # ============ [新增] 初始化安全卫士 ============
    logger.info("Initializing Distance Safety Guard...")
    unique_train_vecs = bo.get_unique_train_vectors()
    safety_guard = DistanceGuard(unique_train_vecs)

    # 6. (可选) 快速评估模型精度
    # 这一步是为了确认模型是否“靠谱”，如果 R2 很低，那后面的推荐也没意义
    logger.info("--- Evaluating Model Accuracy on Test Set ---")

    # 构建测试集特征
    X_test = []
    y_test_true = []

    for item in test_data:
        l_name = item["litmus"]
        if l_name in bo.litmus_to_vector_dict:
            vec = bo.litmus_to_vector_dict[l_name]
            # 注意：RandomForestBO.add 是 param + vec
            X_test.append(list(item["param"]) + list(vec))
            y_test_true.append(item["score"])

    if X_test:
        pred_log = bo.model.predict(np.array(X_test))
        y_test_pred = np.expm1(pred_log)  # 还原

        r2 = r2_score(y_test_true, y_test_pred)
        mae = mean_absolute_error(y_test_true, y_test_pred)
        rho, _ = spearmanr(y_test_true, y_test_pred)

        logger.info(f"Model R^2: {r2:.4f}")
        logger.info(f"Model MAE: {mae:.4f}")
        logger.info(f"Model Rho: {rho:.4f}")
    else:
        logger.warning("No valid test data found (missing vectors?).")

    # 7. 执行稳健参数推荐
    logger.info("=" * 60)

    selector = RobustParamSelector(
        bo.model,
        param_space,
        safety_guard=safety_guard,  # 传入 Guard
        default_param=[0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    )

    recommendations = selector.select_best_params(
        litmus_list=litmus_names_new,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=2.0,  # 提高 alpha，更加厌恶风险 (原来是 1.0)
        dist_threshold=2.0  # 距离阈值：如果是平均距离的2倍以上，就视为"未知测试"
    )

    # 8. 统计与保存结果
    optimized_count = sum(1 for v in recommendations.values() if v['decision'] == 'optimized')
    default_count = len(recommendations) - optimized_count

    logger.info(f"Selection Finished.")
    logger.info(f"  - Optimized: {optimized_count}")
    logger.info(f"  - Default (Safety Fallback): {default_count}")

    output_file = "./make_new_litmus/best_params_recommendation_robust.json"
    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_file}")
    logger.info("=== Done ===")