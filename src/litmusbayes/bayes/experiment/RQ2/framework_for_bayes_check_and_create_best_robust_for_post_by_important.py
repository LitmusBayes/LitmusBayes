import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"

# ================= 配置路径 =================

litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "../RQ3/vector/litmus_vector_two_tower_fixed.log"

litmus_vec_path_new = litmus_vec_path
litmus_path_new = litmus_path
cache_file_path = "../RQ1/cache_norm_final.jsonl"

CLUSTER_JSON_PATH = "./cluster_results_final/cluster_centers.json"
CENTER_LOG_PATH = "./log_record_init_banana_important_norm_final.log.validation_cache.jsonl"


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
                 litmus_vec_path=["/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"]):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            min_samples_leaf=3,
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)

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

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting...")
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    def predict_one(self, litmus_name, param_vec):
        if litmus_name not in self.litmus_to_vector_dict:
            return None
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)
        pred_log = self.model.predict([feature])[0]
        return np.expm1(pred_log)

    def predict_batch(self, litmus_list, param_list):
        X_batch = []
        valid_indices = []

        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]
                X_batch.append(list(param) + list(litmus_vec))
                valid_indices.append(i)

        if not X_batch:
            return [], []

        X_batch_np = np.array(X_batch)
        pred_log = self.model.predict(X_batch_np)
        preds = np.expm1(pred_log)

        return preds, valid_indices


# ================= 新增与修改：构建黑名单与分数映射 =================

def build_cluster_info(cluster_json_path, center_log_path):
    """
    解析聚类文件和中心日志，返回：
    1. litmus_to_center: member -> representative 的映射字典
    2. center_blacklist: representative -> set((p0, p1, p7)) 的黑名单字典
    3. center_combo_scores: representative -> {(p0, p1, p7): log_score} 的组合分数加成字典
    """
    logger = get_logger(LOG_NAME)
    litmus_to_center = {}
    center_blacklist = defaultdict(set)
    center_combo_scores = defaultdict(dict)  # 新增：记录聚类中心在特定组合上的得分

    # 1. 建立成员到中心的映射
    if os.path.exists(cluster_json_path):
        with open(cluster_json_path, 'r') as f:
            clusters = json.load(f)
            for cid, info in clusters.items():
                rep = info["representative"]
                for member in info["members"]:
                    litmus_to_center[member] = rep
        logger.info(f"Loaded cluster mapping for {len(litmus_to_center)} litmus tests.")
    else:
        logger.warning(f"Cluster JSON not found at {cluster_json_path}")

    # 2. 建立中心的参数黑名单与分数映射
    if os.path.exists(center_log_path):
        count = 0
        with open(center_log_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                score = data.get("score", 1.0)
                p = data["param"]
                # 提取 0, 1, 7 位构建特征元组
                combo = (p[0], p[1], p[7])
                litmus_name = data["litmus"]

                # 新增：保存该组合的对数分数 (log1p)，用于后续融合
                center_combo_scores[litmus_name][combo] = np.log1p(score)

                if score < 1.1:
                    center_blacklist[litmus_name].add(combo)
                    count += 1
        logger.info(f"Loaded {count} blacklist rules and mapping scores for cluster centers.")
    else:
        logger.warning(f"Center Log not found at {center_log_path}")

    return litmus_to_center, center_blacklist, center_combo_scores


# ================= 修改后的核心选择类 =================

class RobustParamSelector:
    def __init__(self, model, param_space, litmus_to_center, center_blacklist, center_combo_scores, default_param=None):
        self.model = model
        self.ps = param_space
        self.litmus_to_center = litmus_to_center
        self.center_blacklist = center_blacklist
        self.center_combo_scores = center_combo_scores  # 新增
        self.default_param = default_param if default_param else [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def get_forest_uncertainty(self, X):
        per_tree_pred = [tree.predict(X) for tree in self.model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)

        means = np.mean(per_tree_pred, axis=0)
        stds = np.std(per_tree_pred, axis=0)

        return means, stds

    def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0):
        recommendations = {}

        all_param_vectors = np.array(self.ps.get_all_combinations())

        print(f"Searching best params using Base Model with Filter & Boost (Alpha={alpha})...")

        for litmus in tqdm(litmus_list, desc="Robust Selection (Filtered & Boosted)"):
            if litmus not in litmus_feature_map:
                continue

            # --- 1. 获取黑名单和聚类分数加成 ---
            center = self.litmus_to_center.get(litmus, litmus)  # 找不到中心就默认是自己
            bad_combos = self.center_blacklist.get(center, set())
            combo_scores = self.center_combo_scores.get(center, {})

            valid_indices = []
            bonus_scores = []  # 新增：记录被保留参数的加成分数

            for i, vec in enumerate(all_param_vectors):
                combo = (vec[0], vec[1], vec[7])
                if combo not in bad_combos:
                    valid_indices.append(i)
                    # 如果聚类中心日志里有这个组合的分数，就加上；没有则默认加 0
                    bonus_scores.append(combo_scores.get(combo, 0.0))

            # 如果全部被过滤（极端情况），兜底不抛出异常并设置安全的默认矩阵
            if not valid_indices:
                print(f"litmus test:{litmus} all combinations filtered. Using Default.")
                valid_param_vectors = np.array([self.default_param])
                n_valid_params = 1
                bonus_scores_arr = np.array([0.0])
            else:
                valid_param_vectors = all_param_vectors[valid_indices]
                n_valid_params = len(valid_param_vectors)
                bonus_scores_arr = np.array(bonus_scores)

            # --- 2. 构造输入矩阵 ---
            l_feat = np.array(litmus_feature_map[litmus])
            l_feat_repeated = np.tile(l_feat, (n_valid_params, 1))

            X_batch = np.hstack([valid_param_vectors, l_feat_repeated])

            # --- 3. 获取 Log 空间的均值和方差 ---
            means_log, stds_log = self.get_forest_uncertainty(X_batch)

            # --- 4. 分数融合与稳健惩罚 ---
            # 【核心修改】将聚类中心在 0, 1, 7 位上的表现(对数分数) 叠加到先验预测的对数分数上
            final_means_log = means_log + bonus_scores_arr

            # 使用叠加后的分数计算 LCB（惩罚项依旧是原本的树不确定性）
            robust_scores_log = final_means_log - (alpha * stds_log)
            best_idx = np.argmax(robust_scores_log)

            best_mean_log = means_log[best_idx]  # 记录原始的先验预测得分(用于展示和还原阈值判断)
            best_std_log = stds_log[best_idx]
            best_vec = valid_param_vectors[best_idx]

            # 阈值判断依然基于模型纯粹的预测能力(未加成前的预测)
            predicted_real_score = np.expm1(best_mean_log)

            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = "optimized_with_filter_and_boost"
            else:
                final_vec = self.default_param
                decision_type = "default (low_score)"

            recommendations[litmus] = {
                "param": final_vec,
                "pred_score": float(predicted_real_score),
                "pred_std_log": float(best_std_log),
                "decision": decision_type
            }

        return recommendations


# ================= 主程序 =================

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.robust_eval.log"

    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Robust Evaluation Run (With Filter & Boost) | Seed={SEED} ===")

    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]
    logger.info(f"Found {len(litmus_names)} litmus files.")

    new_litmus_list = get_files(litmus_path_new, '.litmus')
    litmus_names_new = [path.split("/")[-1][:-7] for path in new_litmus_list]

    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=100,
        litmus_vec_path=[litmus_vec_path, litmus_vec_path_new]
    )

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

    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    logger.info(f"Train size: {len(train_data)} | Test size: {len(test_data)}")

    logger.info("Building dataset and fitting model...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])

    t_start = time.time()
    bo.fit()
    logger.info(f"Model training finished in {time.time() - t_start:.2f}s")

    logger.info("--- Evaluating Model Accuracy on Test Set ---")
    X_test = []
    y_test_true = []

    for item in test_data:
        l_name = item["litmus"]
        if l_name in bo.litmus_to_vector_dict:
            vec = bo.litmus_to_vector_dict[l_name]
            X_test.append(list(item["param"]) + list(vec))
            y_test_true.append(item["score"])

    if X_test:
        pred_log = bo.model.predict(np.array(X_test))
        y_test_pred = np.expm1(pred_log)

        r2 = r2_score(y_test_true, y_test_pred)
        mae = mean_absolute_error(y_test_true, y_test_pred)
        rho, _ = spearmanr(y_test_true, y_test_pred)

        logger.info(f"Model R^2: {r2:.4f}")
        logger.info(f"Model MAE: {mae:.4f}")
        logger.info(f"Model Rho: {rho:.4f}")
    else:
        logger.warning("No valid test data found (missing vectors?).")

    logger.info("=" * 60)
    logger.info("Loading cluster maps, blacklists and score boosts...")

    # 提取聚类信息，包含了黑名单以及分数加成
    litmus_to_center, center_blacklist, center_combo_scores = build_cluster_info(CLUSTER_JSON_PATH, CENTER_LOG_PATH)

    logger.info("Starting Robust Parameter Selection with Blacklist Filter & Score Boost...")

    selector = RobustParamSelector(
        model=bo.model,
        param_space=param_space,
        litmus_to_center=litmus_to_center,
        center_blacklist=center_blacklist,
        center_combo_scores=center_combo_scores,  # 传入分数映射
        default_param=[0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    )

    recommendations = selector.select_best_params(
        litmus_list=litmus_names_new,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=2.0
    )

    optimized_count = sum(1 for v in recommendations.values() if v['decision'] == 'optimized_with_filter_and_boost')
    default_count = len(recommendations) - optimized_count

    logger.info(f"Selection Finished.")
    logger.info(f"  - Optimized: {optimized_count}")
    logger.info(f"  - Default (Safety Fallback): {default_count}")

    output_file = "best_params_recommendation_robust_final.json"
    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_file}")
    logger.info("=== Done ===")