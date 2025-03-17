import json
import logging
import os
import random
import time
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
# 保持不变：负责用 50000 条旧数据打底，输入是 [参数11维 + Litmus特征]
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
        if litmus_name not in self.litmus_to_vector_dict: return None
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)
        return self.model.predict([feature])[0]


# ================= 2. 全局残差模型 (Global Residual Model) =================
# 废弃了按 Cluster 训练，现在是一个统一的大模型。特征恢复为 [11维param + Litmus特征]
class GlobalResidualModel:
    def __init__(self, prior_bo: GlobalPriorBO):
        self.prior_bo = prior_bo
        self.logger = get_logger(LOG_NAME)

        self.X = []
        self.y_res = []
        self.raw_score = []
        self.model = None

    def add_new_data(self, litmus_name, param_vec, score):
        prior_pred_log = self.prior_bo.predict_log(litmus_name, param_vec)
        if prior_pred_log is None: return

        litmus_vec = self.prior_bo.litmus_to_vector_dict.get(litmus_name)
        if not litmus_vec: return

        # 计算残差
        true_score_log = np.log1p(score)
        residual = true_score_log - prior_pred_log

        # 💡 核心改变：把 litmus_vec 加回来，让大模型能区分不同的测试环境
        feature = list(param_vec) + list(litmus_vec)

        self.X.append(feature)
        self.y_res.append(residual)
        self.raw_score.append(score)
        if litmus_name == "MP+fence.w.w+ctrl":
            print(f"add new data {litmus_name}, {param_vec}, {score}, {residual}, {true_score_log}, {prior_pred_log}")

    def fit(self, min_samples=10, default_param=None):
        num_samples = len(self.X)
        self.logger.info(f"Start fitting Global Residual Model with {num_samples} new chip samples...")

        if num_samples < min_samples:
            self.logger.warning("Not enough new chip data to train a residual model. Will return 0 residuals.")
            return

        # ================= 1. 提取先验模型的特征重要性 =================
        # 先验模型的前 11 维是参数，后面是 litmus 特征，我们只取前 11 维的参数重要性
        prior_importances = self.prior_bo.model.feature_importances_[:11]
        max_imp = np.max(prior_importances)

        # 将重要性映射为一个 [1.0, 3.0] 的权重放大系数
        # 最重要的参数，权重会被放大 3 倍；最不重要的参数，放大 1 倍 (保持不变)
        if max_imp > 0:
            imp_multipliers = 1.0 + (prior_importances / max_imp) * 2.0
        else:
            imp_multipliers = np.ones(11)

        # 打印一下重要性系数，方便你观察哪些参数被先验模型认为最关键
        self.logger.info(f"Prior Feature Importance Multipliers (1x~3x): {np.round(imp_multipliers, 2)}")

        # ================= 2. 基础的非对称样本加权 =================
        raw_scores = np.array(self.raw_score)
        weights = np.ones(num_samples)

        # 惩罚陷阱与提拔潜力股
        weights[raw_scores < 1.0] = 5.0
        weights[raw_scores > 1.2] = 2.0

        # ================= 3. 💡 基于重要性的动态参数加权 =================
        if default_param is None:
            default_param = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
        default_arr = np.array(default_param)

        # 提取所有样本的前 11 维参数部分
        X_params = np.array(self.X)[:, :11]

        for i, param_vec in enumerate(X_params):
            # 找到当前向量与默认向量不同的位置
            diff_mask = (param_vec != default_arr)
            diff_indices = np.where(diff_mask)[0]

            if len(diff_indices) == 1:
                # 典型的单变量测试：只改变了 1 个参数
                changed_idx = diff_indices[0]
                weights[i] *= imp_multipliers[changed_idx]
            elif len(diff_indices) > 1:
                # 如果是多变量组合测试，取这些发生改变的参数中，最大的重要性系数
                weights[i] *= np.max(imp_multipliers[diff_indices])

        # 因为所有数据合并了，数据量大了，我们可以把树加深，让它有能力防御“缝合怪”
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,  # 💡 深度从 4 放宽到 10
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=SEED
        )

        self.model.fit(np.array(self.X), np.array(self.y_res), sample_weight=weights)
        self.logger.info("Global Residual Model trained successfully.")

    def get_residual_uncertainty(self, X_batch):
        """返回预测的残差均值和标准差"""
        if self.model is None:
            n_samples = len(X_batch)
            return np.zeros(n_samples), np.zeros(n_samples)

        per_tree_pred = [tree.predict(X_batch) for tree in self.model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)

        means_res = np.mean(per_tree_pred, axis=0)
        stds_res = np.std(per_tree_pred, axis=0)
        return means_res, stds_res


# ================= 3. 稳健决策器 (Global Robust Selector) =================
class GlobalRobustSelector:
    def __init__(self, prior_bo: GlobalPriorBO, residual_model: GlobalResidualModel, param_space,
                 default_param=None):
        self.prior_bo = prior_bo
        self.residual_model = residual_model
        self.ps = param_space
        self.default_param = default_param if default_param else [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def select_best_params(self, litmus_list, alpha=1.5, max_mutations=2):
        recommendations = {}

        # 💡 防御防线：限制突变数量，从源头掐断“缝合怪”
        raw_combinations = np.array(self.ps.get_all_combinations())
        default_arr = np.array(self.default_param)
        valid_vectors = []
        for vec in raw_combinations:
            if np.sum(vec != default_arr) <= max_mutations:
                valid_vectors.append(vec)

        all_param_vectors = np.array(valid_vectors)
        n_params = len(all_param_vectors)

        print(f"Searching best params... (Reduced space to {n_params} allowed combinations)")

        PENALTY_FACTOR = 4.0
        BOOST_FACTOR = 2.0

        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            if litmus not in self.prior_bo.litmus_to_vector_dict:
                continue

            l_feat = np.array(self.prior_bo.litmus_to_vector_dict[litmus])
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])

            # 1. 获取先验预测
            prior_preds_log = self.prior_bo.model.predict(X_batch)

            # 2. 获取大残差模型的预测 (现在输入是 11维参数 + Litmus特征)
            means_res_log, stds_res_log = self.residual_model.get_residual_uncertainty(X_batch)

            # 3. 非对称干预
            adjusted_res_log = np.where(
                means_res_log < 0,
                means_res_log * PENALTY_FACTOR,
                means_res_log * BOOST_FACTOR
            )

            # 最终的 Log 期望
            final_means_log = prior_preds_log + adjusted_res_log
            robust_scores_log = final_means_log - (alpha * stds_res_log)

            # 4. 选出最优
            best_idx = np.argmax(robust_scores_log)
            best_mean_log = final_means_log[best_idx]
            best_std_log = stds_res_log[best_idx]
            best_vec = all_param_vectors[best_idx]

            predicted_real_score = np.expm1(best_mean_log)

            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = "optimized_global_residual" if self.residual_model.model else "optimized_prior_only"
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

    def analyze_litmus_params(self, litmus, output_json_path, alpha=1.5):
        print(f"Analyzing all parameter combinations for {litmus}...")
        if litmus not in self.prior_bo.litmus_to_vector_dict: return

        # 分析时为了看全貌，我们不过滤突变数量
        all_param_vectors = np.array(self.ps.get_all_combinations())
        n_params = len(all_param_vectors)

        l_feat = np.array(self.prior_bo.litmus_to_vector_dict[litmus])
        l_feat_repeated = np.tile(l_feat, (n_params, 1))
        X_batch = np.hstack([all_param_vectors, l_feat_repeated])

        prior_preds_log = self.prior_bo.model.predict(X_batch)
        means_res_log, stds_res_log = self.residual_model.get_residual_uncertainty(X_batch)

        PENALTY_FACTOR = 4.0
        BOOST_FACTOR = 2.0
        adjusted_res_log = np.where(means_res_log < 0, means_res_log * PENALTY_FACTOR, means_res_log * BOOST_FACTOR)
        final_means_log = prior_preds_log + adjusted_res_log
        robust_scores_log = final_means_log - (alpha * stds_res_log)

        prior_real = np.expm1(prior_preds_log)
        final_mean_real = np.expm1(final_means_log)
        robust_score_real = np.expm1(robust_scores_log)

        results = []
        for i in range(n_params):
            results.append({
                "param_vec": all_param_vectors[i].tolist(),
                "prior_score_real": float(prior_real[i]),
                "raw_residual_log": float(means_res_log[i]),
                "adjusted_residual_log": float(adjusted_res_log[i]),
                "residual_std_log": float(stds_res_log[i]),
                "expected_mean_real": float(final_mean_real[i]),
                "robust_posterior_score": float(robust_score_real[i])
            })

        results.sort(key=lambda x: x["robust_posterior_score"], reverse=True)

        output_data = {
            "litmus_name": litmus,
            "analysis_params": {"alpha": alpha, "penalty_factor": PENALTY_FACTOR, "boost_factor": BOOST_FACTOR},
            "total_combinations": n_params,
            "ranked_results": results
        }

        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Analysis saved to: {output_json_path}")


# ================= 主程序 =================
if __name__ == "__main__":
    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
    stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
    litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"

    old_cache_file = "../RQ1/cache.jsonl"
    new_chip_file = "log_record_dynamic_cluster_norm.log.validation_cache.jsonl"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.global_residual_eval.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Global Residual Run | Seed={SEED} ===")

    # 1. 训练先验
    prior_bo = GlobalPriorBO(litmus_vec_path)
    if os.path.exists(old_cache_file):
        with open(old_cache_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        prior_bo.add(obj["litmus"], obj["param"], obj["score"])
                    except:
                        pass
    prior_bo.fit()

    # 2. 训练大残差模型 (不再需要 cluster json)
    logger.info("Building Global Residual Model...")
    residual_model = GlobalResidualModel(prior_bo)

    if os.path.exists(new_chip_file):
        with open(new_chip_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        residual_model.add_new_data(obj["litmus"], obj["param"], obj["score"])
                    except:
                        pass

    residual_model.fit()

    # 3. 决策
    logger.info("Starting Global Robust Parameter Selection...")
    full_litmus_list = get_files(litmus_path, '.litmus')
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    param_space = LitmusParamSpace()
    selector = GlobalRobustSelector(prior_bo, residual_model, param_space)

    # 如果你想跑全量预测并保存，解开这里的注释
    # recommendations = selector.select_best_params(litmus_list=litmus_names, alpha=1.5, max_mutations=2)
    # output_file = "best_params_recommendation_global_res.json"
    # with open(output_file, "w") as f:
    #     json.dump(recommendations, f, indent=4)

    # 单独分析那个折磨你的 litmus
    target_litmus = "MP+fence.w.w+ctrl"
    if target_litmus in litmus_names:
        analysis_output_file = f"{target_litmus}_global_param_analysis.json"
        selector.analyze_litmus_params(target_litmus, analysis_output_file, alpha=1.5)