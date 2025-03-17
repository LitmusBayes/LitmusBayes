import json
import logging
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 类定义 =================

class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=200,
                 litmus_vec_path=None):
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
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if not path or not os.path.exists(path):
            return litmus_to_vec
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

    def fit(self):
        self.logger.info(f"Start fitting with {len(self.X)} samples...")
        if not self.X:
            self.logger.warning("No data to fit!")
            return
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

perple_dict_path = "../perple_json"

class RobustParamSelector:
    def __init__(self, model, param_space, default_param=None):
        self.model = model
        self.ps = param_space
        self.default_param = default_param if default_param else [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
        perple_files = get_files(perple_dict_path, "jsonl")
        self.perple_dict_list = {}
        for perple_file in perple_files:
            with open(perple_file, "r") as f:
                loaded_dict = json.load(f)
                self.perple_dict_list[perple_file.split("/")[-1][:-7]] = loaded_dict

    def get_forest_uncertainty(self, X):
        # 批量预测
        per_tree_pred = [tree.predict(X) for tree in self.model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)
        means = np.mean(per_tree_pred, axis=0)
        stds = np.std(per_tree_pred, axis=0)
        return means, stds

    def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0, existing_cache_keys=None):
        """
        existing_cache_keys: set of strings, e.g. "litmus_name|0,2,0,..."
        """
        recommendations = {}

        # 1. 获取全量参数空间

        all_param_vectors = self.ps.get_all_combinations(can_perple=False)  # List[List[int]]
        all_param_vectors_true = self.ps.get_all_combinations(can_perple=True)
        all_param_vectors_np = np.array(all_param_vectors)
        n_params = len(all_param_vectors)

        # 预先生成所有 vector 的字符串形式，用于快速比对 cache
        # 格式必须与 main_iteration 中一致: "p1,p2,p3..."
        all_param_strs = [",".join(map(str, vec)) for vec in all_param_vectors]

        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            if litmus not in litmus_feature_map:
                continue
            # ================= 关键修改：提前过滤已跑过的参数 =================
            valid_indices = []
            if existing_cache_keys:
                # 遍历所有可能的参数组合，检查 "当前litmus|参数" 是否在 cache 中
                # 如果存在，说明跑过了，剔除掉
                # 注意：为了性能，这里尽量使用列表推导或numpy掩码

                # 方法：构建当前 litmus 的所有 key
                # 这样做比循环快：先生成当前litmus的前缀
                prefix = f"{litmus}|"

                # 找出所有未跑过的索引
                valid_indices = [
                    i for i, p_str in enumerate(all_param_strs)
                    if (prefix + p_str) not in existing_cache_keys
                ]
            else:
                valid_indices = list(range(n_params))
            # 如果所有参数都跑完了（极小概率），或者没有有效参数
            if not valid_indices:
                # 策略：如果全部跑完了，就返回默认参数，或者返回历史上最好的（这里为了简单返回默认）
                recommendations[litmus] = {
                    "param": self.default_param,
                    "pred_score": 0.0,
                    "decision": "all_cached_fallback",
                    "is_cached": True  # 标记为已缓存，主程序不跑
                }
                continue

            # 只保留未跑过的向量
            candidate_vectors = all_param_vectors_np[valid_indices]

            # =============================================================

            l_feat = np.array(litmus_feature_map[litmus])
            # 广播特征
            l_feat_repeated = np.tile(l_feat, (len(candidate_vectors), 1))

            # 拼接输入：(Candidate_Params + Litmus_Feature)
            X_batch = np.hstack([candidate_vectors, l_feat_repeated])

            # 预测
            means_log, stds_log = self.get_forest_uncertainty(X_batch)
            # robust_scores_log = means_log - (alpha * stds_log)
            # # 找到最佳索引 (相对于 candidate_vectors 的索引)
            # best_local_idx = np.argmax(robust_scores_log)

            ucb_scores_log = means_log + (alpha * stds_log)
            best_local_idx = np.argmax(ucb_scores_log)

            best_vec = candidate_vectors[best_local_idx]
            best_mean_log = means_log[best_local_idx]

            predicted_real_score = np.expm1(best_mean_log)
            final_vec = []
            decision_type = ""

            # 阈值判断
            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = "optimized"
            else:
                final_vec = self.default_param
                decision_type = "default (low_score)"

            # double check: 理论上这里肯定不会是 cached 的了，除非 default_param 被强行选中且已跑过
            is_cached = False
            if existing_cache_keys:
                check_key = f"{litmus}|" + ",".join(map(str, final_vec))
                if check_key in existing_cache_keys:
                    is_cached = True

            recommendations[litmus] = {
                "param": final_vec,
                "pred_score": float(predicted_real_score),
                "decision": decision_type,
                "is_cached": is_cached
            }

        return recommendations


# ================= 封装后的入口函数 =================

def main_optimizer(litmus_path, cache_file_path, litmus_vec_path, output_json_path, log_base_path):
    random.seed(SEED)
    np.random.seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        log_file=f"{log_base_path}.{ts}.optimizer.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== [Optimizer] Start | Seed={SEED} ===")

    # 1. 读取 Litmus 列表
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]
    logger.info(f"Found {len(litmus_names)} litmus files.")

    # 2. 初始化
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=200,
        litmus_vec_path=litmus_vec_path
    )

    # 3. 加载训练数据 & 构建 Cache Key 集合
    logger.info(f"Loading training data from {cache_file_path} ...")
    all_data = []
    existing_cache_keys = set()

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        all_data.append(obj)
                        # 构建 key: litmus|p1,p2...
                        # 确保这里的 param 是 list 且转为 str 格式统一
                        key = f"{obj['litmus']}|" + ",".join(map(str, obj['param']))
                        existing_cache_keys.add(key)
                    except:
                        pass
    else:
        logger.warning("Cache file not found! Training with empty data.")

    logger.info(f"Loaded {len(all_data)} records, {len(existing_cache_keys)} unique cached keys.")

    if len(all_data) > 0:
        for item in all_data:
            bo.add(item["litmus"], item["param"], item["score"])

        t_start = time.time()
        bo.fit()
        logger.info(f"Model fitted in {time.time() - t_start:.2f}s")
    else:
        logger.warning("No data to fit. Using random/default.")

    # 4. 推荐
    logger.info("Starting Robust Parameter Selection (Filtering Cached Params)...")
    selector = RobustParamSelector(bo.model, param_space)

    # 传入 existing_cache_keys，在内部直接排除掉
    recommendations = selector.select_best_params(
        litmus_list=litmus_names,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=1.0,
        existing_cache_keys=existing_cache_keys
    )

    # 5. 保存
    optimized_count = sum(1 for v in recommendations.values() if v['decision'] == 'optimized')
    # 由于我们在内部已经排除了 cache，这里 theoretically is_cached 应该大部分为 False
    # 除非 fallback 到了 default 且 default 刚好跑过
    cached_fallback_count = sum(1 for v in recommendations.values() if v['is_cached'])

    logger.info(f"Optimized (New): {optimized_count}, Fallback to Cached Default: {cached_fallback_count}")

    with open(output_json_path, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_json_path}")
    return recommendations