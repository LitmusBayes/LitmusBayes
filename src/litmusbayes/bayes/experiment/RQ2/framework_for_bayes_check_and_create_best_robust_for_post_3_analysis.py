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
# 保持不变：负责用 50000 条旧数据打底，提供基础得分的保障
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


# ================= 2. OFAT 独立残差模型 (彻底抛弃随机森林组合幻觉) =================
class OFATResidualModel:
    def __init__(self, prior_bo: GlobalPriorBO, default_param=None):
        self.prior_bo = prior_bo
        self.default_param = np.array(default_param if default_param else [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0])
        self.logger = get_logger(LOG_NAME)

        # 记录单点突变的残差和真实得分
        # 格式: {(param_index, param_value): {"residual": float, "raw_score": float}}
        self.mutation_memory = {}

    def add_new_data(self, litmus_name, param_vec, score):
        prior_pred_log = self.prior_bo.predict_log(litmus_name, param_vec)
        if prior_pred_log is None: return

        true_score_log = np.log1p(score)
        residual = true_score_log - prior_pred_log

        # 寻找当前向量和默认向量的差异
        vec = np.array(param_vec)
        diff_indices = np.where(vec != self.default_param)[0]

        # 💡 只记录纯粹的单点突变 (OFAT) 数据
        if len(diff_indices) == 1:
            idx = diff_indices[0]
            val = vec[idx]
            key = (idx, val)

            # 全局最悲观原则：如果同一个突变测了多次（在不同用例上），我们保留得分最低的那次！
            # 这是最安全的防御，只要它在某处是 0.0 的毒药，全局禁用。
            if key not in self.mutation_memory or score < self.mutation_memory[key]["raw_score"]:
                self.mutation_memory[key] = {
                    "residual": residual,
                    "raw_score": score
                }

    def fit(self, *args, **kwargs):
        # 字典模型不需要复杂的 fit 过程，直接汇报学到了多少条独立经验
        self.logger.info(f"OFAT Residual Model built with {len(self.mutation_memory)} independent mutation rules.")
        for k, v in self.mutation_memory.items():
            self.logger.info(
                f"Learned Rule -> Param[{k[0]}]={k[1]}: Res={v['residual']:+.4f}, Min_Score={v['raw_score']}")

    def get_adjusted_residual(self, param_vec_batch, penalty_factor=4.0, boost_factor=2.0):
        """
        直接根据独立的突变累加残差，并执行一票否决
        """
        n_samples = len(param_vec_batch)
        final_residuals = np.zeros(n_samples)

        for i in range(n_samples):
            vec = param_vec_batch[i]
            diff_indices = np.where(vec != self.default_param)[0]

            total_res = 0.0
            is_poisoned = False

            for idx in diff_indices:
                val = vec[idx]
                key = (idx, val)

                if key in self.mutation_memory:
                    mem = self.mutation_memory[key]

                    # 💡 绝对的物理隔离：如果日志里这个参数得分是 0.1 以下的毒药
                    if mem["raw_score"] < 0.1:
                        is_poisoned = True
                        break  # 不用再算了，这组参数直接判死刑

                    # 依然保留你的非对称干预（重拳出击与提拔潜力股）
                    res = mem["residual"]
                    if res < 0:
                        total_res += res * penalty_factor
                    else:
                        total_res += res * boost_factor
                else:
                    # 遇到了在新芯片上完全没测过的突变：给一点微小的惩罚防止模型滥用没见过的数据
                    # total_res -= 0.05
                    pass

                    # 如果触发了毒药，给一个极其巨大的负残差（保证最终得分极低）
            if is_poisoned:
                final_residuals[i] = -100.0
            else:
                final_residuals[i] = total_res

        return final_residuals


# ================= 3. 稳健决策器 (OFAT Robust Selector) =================
class GlobalRobustSelector:
    def __init__(self, prior_bo: GlobalPriorBO, residual_model: OFATResidualModel, param_space,
                 default_param=None):
        self.prior_bo = prior_bo
        self.residual_model = residual_model
        self.ps = param_space
        self.default_param = default_param if default_param else [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def select_best_params(self, litmus_list, max_mutations=2):
        recommendations = {}

        # 💡 物理隔离：只生成最多包含 N 个突变的参数组合，彻底掐断“8变异缝合怪”
        raw_combinations = np.array(self.ps.get_all_combinations())
        default_arr = np.array(self.default_param)
        valid_vectors = []
        for vec in raw_combinations:
            if np.sum(vec != default_arr) <= max_mutations:
                valid_vectors.append(vec)

        all_param_vectors = np.array(valid_vectors)
        n_params = len(all_param_vectors)

        print(
            f"Searching best params... (Reduced space to {n_params} allowed combinations, max {max_mutations} mutations)")

        PENALTY_FACTOR = 4.0
        BOOST_FACTOR = 2.0

        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            if litmus not in self.prior_bo.litmus_to_vector_dict:
                continue

            # 1. 获取先验预测
            l_feat = np.array(self.prior_bo.litmus_to_vector_dict[litmus])
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])
            prior_preds_log = self.prior_bo.model.predict(X_batch)

            # 2. 获取 OFAT 独立残差预测 (自带一票否决和非对称放大)
            adjusted_res_log = self.residual_model.get_adjusted_residual(
                all_param_vectors, penalty_factor=PENALTY_FACTOR, boost_factor=BOOST_FACTOR
            )

            # 3. 最终的 Log 期望 = 先验打底 + 修正后的残差总和
            # 因为抛弃了随机森林，不再需要 alpha 惩罚方差
            robust_scores_log = prior_preds_log + adjusted_res_log

            # 4. 选出最优
            best_idx = np.argmax(robust_scores_log)
            best_mean_log = robust_scores_log[best_idx]
            best_vec = all_param_vectors[best_idx]

            predicted_real_score = np.expm1(best_mean_log)

            if predicted_real_score > 1.0 and adjusted_res_log[best_idx] > -50:
                final_vec = best_vec.tolist()
                decision_type = "optimized_ofat_residual"
            else:
                final_vec = self.default_param
                decision_type = "default (safety fallback)"

            recommendations[litmus] = {
                "param": final_vec,
                "pred_score": float(predicted_real_score),
                "decision": decision_type
            }

        return recommendations

    def analyze_litmus_params(self, litmus, output_json_path, max_mutations=2):
        print(f"Analyzing parameter combinations for {litmus} (Max Mutations: {max_mutations})...")
        if litmus not in self.prior_bo.litmus_to_vector_dict: return

        # 分析时也严格过滤，保持 JSON 干净
        raw_combinations = np.array(self.ps.get_all_combinations())
        default_arr = np.array(self.default_param)
        valid_vectors = []
        for vec in raw_combinations:
            if np.sum(vec != default_arr) <= max_mutations:
                valid_vectors.append(vec)

        all_param_vectors = np.array(valid_vectors)
        n_params = len(all_param_vectors)

        l_feat = np.array(self.prior_bo.litmus_to_vector_dict[litmus])
        l_feat_repeated = np.tile(l_feat, (n_params, 1))
        X_batch = np.hstack([all_param_vectors, l_feat_repeated])

        prior_preds_log = self.prior_bo.model.predict(X_batch)

        PENALTY_FACTOR = 4.0
        BOOST_FACTOR = 2.0
        adjusted_res_log = self.residual_model.get_adjusted_residual(
            all_param_vectors, penalty_factor=PENALTY_FACTOR, boost_factor=BOOST_FACTOR
        )

        robust_scores_log = prior_preds_log + adjusted_res_log

        prior_real = np.expm1(prior_preds_log)
        robust_score_real = np.expm1(robust_scores_log)

        results = []
        for i in range(n_params):
            # 过滤掉被“毒药参数一票否决”导致得分低于 -50 的垃圾组合
            if adjusted_res_log[i] < -50:
                continue

            results.append({
                "param_vec": all_param_vectors[i].tolist(),
                "prior_score_real": float(prior_real[i]),
                "adjusted_residual_log": float(adjusted_res_log[i]),
                "robust_posterior_score": float(robust_score_real[i])
            })

        # 按最终得分降序排列
        results.sort(key=lambda x: x["robust_posterior_score"], reverse=True)

        output_data = {
            "litmus_name": litmus,
            "analysis_params": {"max_mutations": max_mutations, "penalty_factor": PENALTY_FACTOR,
                                "boost_factor": BOOST_FACTOR},
            "total_combinations_analyzed": len(results),
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
    log_file_name = f"{stat_log_base}.{ts}.ofat_residual_eval.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start OFAT Residual Run | Seed={SEED} ===")

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

    # 2. 训练 OFAT 独立残差模型
    logger.info("Building OFAT Residual Model...")
    residual_model = OFATResidualModel(prior_bo)

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

    # ----------------------------------------------------
    # 如果你想跑全量预测并生成推荐 JSON，解开下面的注释
    # ----------------------------------------------------
    recommendations = selector.select_best_params(litmus_list=litmus_names, max_mutations=2)
    output_file = "best_params_recommendation_ofat_res.json"
    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)
    logger.info(f"Recommendations saved to: {output_file}")

    # ----------------------------------------------------
    # 单独分析那个折磨你的 litmus (生成分析报告)
    # ----------------------------------------------------
    target_litmus = "MP+fence.w.w+ctrl"
    if target_litmus in litmus_names:
        analysis_output_file = f"{target_litmus}_ofat_param_analysis.json"
        # 💡 这里限制 max_mutations=2，彻底阻绝缝合怪
        selector.analyze_litmus_params(target_litmus, analysis_output_file, max_mutations=2)
    else:
        logger.warning(f"Target litmus {target_litmus} not found in the list.")