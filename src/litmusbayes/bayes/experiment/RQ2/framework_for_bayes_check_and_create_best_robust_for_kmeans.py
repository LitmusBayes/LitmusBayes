import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 1. 先验模型 (Global Prior BO) =================
# 这个模型负责用 50000 条旧数据打底，输入是 [参数11维 + Litmus特征]
class GlobalPriorBO:
    def __init__(self, litmus_vec_path):
        self.model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=-1,
            min_samples_leaf=3,
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.litmus_to_vector_dict = self._load_litmus_vectors(litmus_vec_path)
        self.logger = get_logger(LOG_NAME)

    def _load_litmus_vectors(self, path):
        litmus_to_vec = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line: continue
                name, vec_str = line.split(":", 1)
                litmus_to_vec[name] = list(eval(vec_str))
        return litmus_to_vec

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict: return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting Prior Model with {len(self.X)} old samples...")
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    def predict_log(self, litmus_name, param_vec):
        """只返回 Log 空间的预测值，供残差计算使用"""
        if litmus_name not in self.litmus_to_vector_dict: return None
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)
        return self.model.predict([feature])[0]


# ================= 2. 聚类残差群组 (Cluster Residual Ensemble) =================
# 这个类管理多个 RF，每个 Cluster 一个。特征仅为 11 维的 param。
class ClusterResidualEnsemble:
    def __init__(self, prior_bo: GlobalPriorBO, cluster_json_path):
        self.prior_bo = prior_bo
        self.logger = get_logger(LOG_NAME)

        # 解析 Cluster 映射：litmus_name -> cluster_id
        self.litmus_to_cluster = {}
        with open(cluster_json_path, 'r') as f:
            cluster_data = json.load(f)
            for cid, info in cluster_data.items():
                for member in info["members"]:
                    self.litmus_to_cluster[member] = str(cid)

        # 存储每个 Cluster 的训练数据
        self.cluster_train_data = defaultdict(lambda: {"X": [], "y_res": []})
        # 存储训练好的 Cluster 模型
        self.cluster_models = {}

    def add_new_data(self, litmus_name, param_vec, score):
        cid = self.litmus_to_cluster.get(litmus_name)
        if not cid: return

        prior_pred_log = self.prior_bo.predict_log(litmus_name, param_vec)
        if prior_pred_log is None: return

        # 计算残差：真实的 Log 得分 - 先验 Log 预测值
        true_score_log = np.log1p(score)
        residual = true_score_log - prior_pred_log

        # 注意：这里的 X 只有 param_vec，维度大幅降低！
        self.cluster_train_data[cid]["X"].append(list(param_vec))
        self.cluster_train_data[cid]["y_res"].append(residual)
        print(f"litmus: {litmus_name}, score: {score}, param: {param_vec}, residual: {residual}")

    def fit(self, min_samples=5):
        self.logger.info("Start fitting Cluster-specific Residual Models...")
        trained_clusters = 0
        skipped_clusters = 0

        for cid, data in self.cluster_train_data.items():
            num_samples = len(data["X"])
            if num_samples < min_samples:
                # 样本太少，放弃训练，推理时残差直接视为 0
                self.logger.warning(f"Cluster {cid} has only {num_samples} samples. Skipping residual training.")
                skipped_clusters += 1
                continue

            # 针对小样本设置保守的 RF
            rf = RandomForestRegressor(
                n_estimators=50,  # 树不用太多
                max_depth=10,  # 极浅的树，防止强行记忆 400 条数据
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=SEED
            )
            rf.fit(np.array(data["X"]), np.array(data["y_res"]))
            self.cluster_models[cid] = rf
            trained_clusters += 1
            self.logger.info(f"Cluster {cid} residual model trained with {num_samples} samples.")

        self.logger.info(f"Residual Models built: {trained_clusters}, Skipped: {skipped_clusters}")

    def get_residual_uncertainty(self, cid, X_params_batch):
        """返回特定 Cluster 的残差均值和标准差"""
        if cid not in self.cluster_models:
            # 如果该 Cluster 没有残差模型（数据太少），退化为 0 残差，0 方差
            n_samples = len(X_params_batch)
            return np.zeros(n_samples), np.zeros(n_samples)

        model = self.cluster_models[cid]
        per_tree_pred = [tree.predict(X_params_batch) for tree in model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)

        means_res = np.mean(per_tree_pred, axis=0)
        stds_res = np.std(per_tree_pred, axis=0)
        return means_res, stds_res


# ================= 3. 稳健决策器 =================
class ClusterRobustSelector:
    def __init__(self, prior_bo: GlobalPriorBO, residual_ensemble: ClusterResidualEnsemble, param_space,
                 default_param=None):
        self.prior_bo = prior_bo
        self.residual_ensemble = residual_ensemble
        self.ps = param_space
        # 按照你的要求，更新了 default_param
        self.default_param = default_param if default_param else [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def select_best_params(self, litmus_list, alpha=1.5):
        recommendations = {}
        all_param_vectors = np.array(self.ps.get_all_combinations())
        n_params = len(all_param_vectors)

        print(f"Searching best params using Cluster Residuals (Alpha={alpha})...")

        for litmus in tqdm(litmus_list, desc="Robust Cluster Selection"):
            cid = self.residual_ensemble.litmus_to_cluster.get(litmus)
            # 基础检查：是否有 Litmus 向量
            if litmus not in self.prior_bo.litmus_to_vector_dict:
                continue

            # 1. 获取先验模型的批量预测
            l_feat = np.array(self.prior_bo.litmus_to_vector_dict[litmus])
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_prior_batch = np.hstack([all_param_vectors, l_feat_repeated])

            prior_preds_log = self.prior_bo.model.predict(X_prior_batch)

            # 2. 获取聚类残差模型的批量预测 (如果 cid 不存在，这里会自动返回全 0)
            means_res_log, stds_res_log = self.residual_ensemble.get_residual_uncertainty(cid, all_param_vectors)

            # 3. 合并与 LCB
            final_means_log = prior_preds_log + 2 * means_res_log
            # robust_scores_log = final_means_log - (alpha * stds_res_log)
            robust_scores_log = final_means_log

            # 4. 选出最优
            best_idx = np.argmax(robust_scores_log)
            best_mean_log = final_means_log[best_idx]
            best_std_log = stds_res_log[best_idx]
            best_vec = all_param_vectors[best_idx]

            predicted_real_score = np.expm1(best_mean_log)

            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = f"optimized_cluster_{cid}" if cid in self.residual_ensemble.cluster_models else "optimized_prior_only"
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
    # 【新增】你的聚类 JSON 文件路径
    cluster_json_file = "./cluster_results_final/cluster_centers.json"  # 请替换为你的真实路径

    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
    stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
    # litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
    litmus_vec_path = "../RQ3/vector/litmus_vector_two_tower_fixed.log"

    litmus_vec_path_new = litmus_vec_path
    litmus_path_new = litmus_path
    old_cache_file = "../RQ1/cache_norm_final.jsonl"
    new_chip_file = "log_record_init_banana_important_norm_final.log.validation_cache.jsonl"
    output_file = "best_params_recommendation_robust_final.json"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.cluster_residual_eval.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Cluster-based Residual Run | Seed={SEED} ===")

    # 1. 初始化并训练全局先验模型
    prior_bo = GlobalPriorBO(litmus_vec_path)

    logger.info(f"Loading OLD data from {old_cache_file} ...")
    if os.path.exists(old_cache_file):
        with open(old_cache_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        prior_bo.add(obj["litmus"], obj["param"], obj["score"])
                    except:
                        pass

    t_start = time.time()
    prior_bo.fit()
    logger.info(f"Prior Model training finished in {time.time() - t_start:.2f}s")

    # 2. 初始化并训练聚类残差群组
    logger.info("=" * 60)
    logger.info("Building Cluster Residual Ensemble...")
    residual_ensemble = ClusterResidualEnsemble(prior_bo, cluster_json_file)

    if os.path.exists(new_chip_file):
        with open(new_chip_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        residual_ensemble.add_new_data(obj["litmus"], obj["param"], obj["score"])
                    except:
                        pass

    # 过滤极小样本 cluster 并训练
    residual_ensemble.fit(min_samples=5)

    # 3. 稳健推断与选择
    logger.info("=" * 60)
    logger.info("Starting Robust Parameter Selection...")

    # full_litmus_list = get_files(litmus_path, '.litmus')
    # litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]
    litmus_names = []
    filter_path = "./a_important_litmus.txt"
    with open(filter_path, "r") as f:
        filter_litmus_list = []
        lines = f.readlines()
        for line in lines:
            litmus_names.append(line.strip().split("|")[0].strip())

    param_space = LitmusParamSpace()
    selector = ClusterRobustSelector(prior_bo, residual_ensemble, param_space)

    recommendations = selector.select_best_params(
        litmus_list=litmus_names,
        alpha=1.5  # 同样推荐 1.5 左右的惩罚力度
    )

    # 4. 统计结果
    opt_cluster = sum(1 for v in recommendations.values() if "optimized_cluster" in v['decision'])
    opt_prior = sum(1 for v in recommendations.values() if v['decision'] == 'optimized_prior_only')
    default_cnt = sum(1 for v in recommendations.values() if "default" in v['decision'])

    logger.info(f"Selection Finished.")
    logger.info(f"  - Optimized (Hit Cluster Residual): {opt_cluster}")
    logger.info(f"  - Optimized (Fell back to Prior): {opt_prior}")
    logger.info(f"  - Default (Safety Fallback): {default_cnt}")

    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_file}")