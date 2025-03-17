import json
import logging
import os
import random
import time
import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================= 原有项目引用 =================
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 1. 基础类定义 =================

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

        # 加载向量
        self.litmus_to_vector_dict = {}
        for vec_path in litmus_vec_path:
            litmus_vec_dict = self.load_litmus_vectors(vec_path)
            for vec in litmus_vec_dict:
                self.litmus_to_vector_dict[vec] = litmus_vec_dict[vec]

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if not os.path.exists(path):
            return litmus_to_vec
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line: continue
                name, vec_str = line.split(":", 1)
                vec = eval(vec_str)
                litmus_to_vec[name.strip()] = list(vec)
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


class RobustParamSelector:
    def __init__(self, model, param_space, default_param=None):
        self.model = model
        self.ps = param_space
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
        n_params = len(all_param_vectors)

        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            if litmus not in litmus_feature_map:
                continue

            l_feat = np.array(litmus_feature_map[litmus])
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])

            means_log, stds_log = self.get_forest_uncertainty(X_batch)
            robust_scores_log = means_log - (alpha * stds_log)
            best_idx = np.argmax(robust_scores_log)

            best_mean_log = means_log[best_idx]
            best_std_log = stds_log[best_idx]
            best_vec = all_param_vectors[best_idx]

            predicted_real_score = np.expm1(best_mean_log)

            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = "optimized"
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


# ================= 2. 核心新增：跨平台迁移分析模块 =================

def analyze_transferable_params(bo, param_space, top_k=3, sample_size=5000):
    """
    分析对性能影响最大的 Top K 参数，并评估它们各自的最佳取值以及组合的笛卡尔积。
    """
    logger = bo.logger
    logger.info("=" * 60)
    logger.info(f"🚀 [Migration Analysis] 正在提取最重要的前 {top_k} 个参数...")

    # 1. 确定参数维度
    # 通过训练数据的第一个样本获取参数的长度
    if not bo.X:
        logger.error("No training data found in BO model.")
        return [], []

    param_dim = len(param_space.get_all_combinations()[0])

    # 2. 获取特征重要性并排序
    importances = bo.model.feature_importances_[:param_dim]
    top_k_indices = np.argsort(importances)[-top_k:][::-1]

    logger.info("--- 全局参数重要性排名 (Top 3) ---")
    for rank, idx in enumerate(top_k_indices):
        logger.info(f"  Rank {rank + 1} -> Param Index [{idx}]: Importance = {importances[idx]:.4f}")

    # 3. 获取所有合法参数组合，用于提取 Top K 参数的独立合法候选值
    all_param_vectors = np.array(param_space.get_all_combinations())
    unique_vals_per_param = [np.unique(all_param_vectors[:, idx]) for idx in top_k_indices]

    # 4. 准备背景评估数据集 (部分依赖分析 Partial Dependence)
    X_bg = np.array(bo.X)
    if len(X_bg) > sample_size:
        np.random.seed(SEED)
        indices = np.random.choice(len(X_bg), sample_size, replace=False)
        X_bg = X_bg[indices]

    # 5. 分析【每个参数的独立最佳取值】
    logger.info("-" * 40)
    logger.info("📍 分析每个重要参数的【最佳单点取值】...")
    best_single_values = {}

    for i, param_idx in enumerate(top_k_indices):
        vals = unique_vals_per_param[i]
        val_scores = []
        for val in vals:
            X_eval = X_bg.copy()
            X_eval[:, param_idx] = val  # 将该参数强制设为指定值
            pred_log = bo.model.predict(X_eval)
            mean_score = np.mean(np.expm1(pred_log))
            val_scores.append((val, mean_score))

        # 按得分排序，找到该参数的最佳值
        val_scores.sort(key=lambda x: x[1], reverse=True)
        best_single_values[int(param_idx)] = val_scores

        logger.info(f"  Param Index [{param_idx}] 的取值表现排名:")
        for rank, (v, score) in enumerate(val_scores):
            logger.info(f"    - 取值 {int(v)}: 预期全局均分 = {score:.4f} {'(最佳🏆)' if rank == 0 else ''}")

    # 6. 分析【笛卡尔积组合的最佳取值】
    logger.info("-" * 40)
    logger.info(f"🧬 分析前 {top_k} 个参数的【笛卡尔积组合】潜力...")
    top_k_combinations = list(itertools.product(*unique_vals_per_param))
    logger.info(f"共生成 {len(top_k_combinations)} 种组合，开始评估...")

    combo_results = []
    for combo in tqdm(top_k_combinations, desc="Evaluating Cartesian Product"):
        X_eval = X_bg.copy()
        for i, param_idx in enumerate(top_k_indices):
            X_eval[:, param_idx] = combo[i]

        pred_log = bo.model.predict(X_eval)
        mean_score = np.mean(np.expm1(pred_log))
        combo_results.append({
            "combo": [int(x) for x in combo],
            "expected_score": float(mean_score)
        })

    combo_results.sort(key=lambda x: x["expected_score"], reverse=True)

    logger.info("--- 🌟 跨平台迁移：最强组合 Top 5 🌟 ---")
    logger.info(f"组合格式: (Param[{top_k_indices[0]}], Param[{top_k_indices[1]}], Param[{top_k_indices[2]}])")
    for i in range(min(5, len(combo_results))):
        res = combo_results[i]
        logger.info(f"  Top {i + 1}: 组合 {res['combo']} -> 预期加速比: {res['expected_score']: .4f}")

    return top_k_indices, best_single_values, combo_results


# ================= 3. 主程序 =================

if __name__ == "__main__":
    # 路径配置
    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
    stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
    litmus_vec_path = "../RQ3/vector/litmus_vector_two_tower_fixed.log"
    litmus_vec_path_new = litmus_vec_path
    litmus_path_new = litmus_path
    cache_file_path = "../RQ1/cache_norm_final.jsonl"

    # 初始化设置
    random.seed(SEED)
    np.random.seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.robust_eval.log"

    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Evaluation & Migration Prior Generation | Seed={SEED} ===")

    # 准备 Litmus 列表
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    new_litmus_list = get_files(litmus_path_new, '.litmus')
    litmus_names_new = [path.split("/")[-1][:-7] for path in new_litmus_list]

    # 初始化 BO
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=100,
        litmus_vec_path=[litmus_vec_path, litmus_vec_path_new]
    )

    # 加载数据
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass

    logger.info(f"Total records loaded: {len(all_data)}")

    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # 训练模型
    logger.info("Building dataset and fitting model...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])

    t_start = time.time()
    bo.fit()
    logger.info(f"Model training finished in {time.time() - t_start:.2f}s")

    # ==========================================================
    # 🎯 核心调用：执行跨平台迁移参数分析
    # ==========================================================
    top_params, best_singles, top_combos = analyze_transferable_params(bo, param_space, top_k=3)

    # 将先验结果保存下来，供新平台热启动使用
    migration_priors = {
        "top_param_indices": top_params.tolist(),
        "best_single_values": {str(k): [{"value": float(v), "score": float(s)} for v, s in vals] for k, vals in
                               best_singles.items()},
        "best_combinations": top_combos
    }
    prior_out_path = "migration_priors_top3.json"
    with open(prior_out_path, "w") as f:
        json.dump(migration_priors, f, indent=4)
    logger.info(f"✅ Migration priors saved to: {prior_out_path}")
    # ==========================================================

    # 原有的稳健推荐流程 (可选保留，用于当前平台的输出)
    logger.info("=" * 60)
    logger.info("Starting Robust Parameter Selection for Current Platform...")
    selector = RobustParamSelector(bo.model, param_space, default_param=[0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0])

    recommendations = selector.select_best_params(
        litmus_list=litmus_names_new,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=1.0
    )

    optimized_count = sum(1 for v in recommendations.values() if v['decision'] == 'optimized')
    default_count = len(recommendations) - optimized_count

    logger.info(f"Selection Finished. Optimized: {optimized_count}, Default: {default_count}")

    output_file = "best_params_recommendation_robust.json"
    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_file}")
    logger.info("=== Done ===")