import json
import logging
import os
import random
import time
from collections import defaultdict
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

# 引入你的自定义库
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"

# ================= 配置路径 =================
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
# litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
litmus_vec_path = "../RQ3/vector/litmus_vector_two_tower_fixed.log"
litmus_vec_path_new = litmus_vec_path
cache_file_path = "../RQ1/cache_norm_final.jsonl"

# 【新增】存放你想要采样的 litmus test 名字的 txt 文件路径
target_litmus_txt = "./kmeans_litmus_final.txt"
# 【新增】输出最终参数结果的 json 文件路径
output_json_path = "./diverse_selected_params_final.json"
CLUSTER_JSON_PATH = './cluster_results_final/cluster_centers.json'


# ================= 类定义 (保留你原有的 BO 逻辑) =================

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
        self.f = open(path, "a", encoding="utf-8")

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        pass


class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=100,
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


class RobustTopKParamSelector:
    def __init__(self, model, param_space, default_param=None):
        """
        :param model: 训练好的 sklearn RandomForestRegressor
        :param param_space: LitmusParamSpace 实例
        :param default_param: 兜底参数，默认 [0,1,0,0,0,0,2,0,0,0,0]
        """
        self.model = model
        self.ps = param_space
        self.default_param = default_param if default_param else [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def get_forest_uncertainty(self, X):
        """获取随机森林中每棵树的预测，计算 Mean 和 Std (Log 空间)"""
        per_tree_pred = [tree.predict(X) for tree in self.model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)

        means = np.mean(per_tree_pred, axis=0)
        stds = np.std(per_tree_pred, axis=0)

        return means, stds

    def select_top_k_params(self, litmus_list, litmus_feature_map, alpha=1.0, k=10):
        """
        核心逻辑：Score_Log = Mean_Log - (alpha * Std_Log)
        根据稳健分数排序，提取前 K 个最佳参数。
        """
        recommendations = {}

        # 1. 获取全量参数空间
        all_param_vectors = np.array(self.ps.get_all_combinations())
        n_params = len(all_param_vectors)

        print(f"Searching Top-{k} robust params from {n_params} combinations per litmus (Alpha={alpha})...")

        for litmus in tqdm(litmus_list, desc="Robust Top-K Selection"):
            if litmus not in litmus_feature_map:
                continue

            l_feat = np.array(litmus_feature_map[litmus])

            # 2. 构造输入矩阵
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])

            # 3. 获取 Log 空间的均值和方差
            means_log, stds_log = self.get_forest_uncertainty(X_batch)

            # 4. 计算稳健分数 (惩罚不确定性)
            robust_scores_log = means_log - (alpha * stds_log)

            # 5. 获取前 K 个稳健分数最高的索引 (升序排列后取最后 k 个并翻转)
            top_indices = np.argsort(robust_scores_log)[-k:][::-1]

            # 6. 整理 Top-K 结果
            for rank, idx in enumerate(top_indices):
                best_mean_log = means_log[idx]
                best_std_log = stds_log[idx]
                best_vec = all_param_vectors[idx]

                # 还原真实预测分数
                predicted_real_score = np.expm1(best_mean_log)

                # 阈值判断逻辑 (与你原代码保持一致)
                if predicted_real_score > 1.0:
                    final_vec = best_vec.tolist()
                    decision_type = f"optimized_rank_{rank}"
                else:
                    final_vec = self.default_param
                    decision_type = f"default_fallback_rank_{rank}"

                result_key = f"{litmus}_rank_{rank}"
                recommendations[result_key] = {
                    "param": final_vec,
                    "pred_score": float(predicted_real_score),
                    "pred_std_log": float(best_std_log),
                    "robust_score_log": float(robust_scores_log[idx]), # 记录排序依据的得分
                    "rank": rank,
                    "decision": decision_type
                }

        return recommendations

# ================= 主程序 =================

if __name__ == "__main__":
    # 1. 基础设置
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.diverse_eval.log"

    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Robust Top-K Parameter Selection | Seed={SEED} ===")

    # 假设你在顶部定义了 target_json_path = "./kmeans_clusters.json"
    # 如果没有定义，请确保在这里或全局配置里定义好它
    target_json_path = CLUSTER_JSON_PATH # <-- 请替换为你的实际 json 路径

    # 2. 从 JSON 文件读取 target litmus names (提取 representative)
    if not os.path.exists(target_json_path):
        logger.error(f"Target JSON list {target_json_path} not found! Please check the path.")
        exit(1)

    logger.info(f"Reading target litmus tests from JSON: {target_json_path}")
    target_litmus_names = []
    with open(target_json_path, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
        # 遍历 JSON 里的每一个 key (如 "15")，提取其中的 representative
        for cluster_id, cluster_info in cluster_data.items():
            if "representative" in cluster_info:
                target_litmus_names.append(cluster_info["representative"])

    logger.info(f"Loaded {len(target_litmus_names)} representative target litmus tests to sample.")

    # 读取全局作为训练背景的 Litmus 列表
    full_litmus_list = get_files(litmus_path)
    all_litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO 和 参数空间
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        all_litmus_names,
        n_estimators=100,
        litmus_vec_path=[litmus_vec_path, litmus_vec_path_new]
    )

    # 4. 加载训练数据 (Cache)
    logger.info(f"Loading training data from {cache_file_path} ...")
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r", encoding="utf-8") as f:
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

    # 5. 训练模型 (这里全量数据直接训练，不再切分测试集)
    logger.info("Building dataset and fitting model...")
    for item in all_data:
        bo.add(item["litmus"], item["param"], item["score"])

    t_start = time.time()
    bo.fit()
    logger.info(f"Model training finished in {time.time() - t_start:.2f}s")

    # 7. 执行稳健参数推荐 (Robust Top-K)
    logger.info("=" * 60)
    logger.info("Starting Robust Top-K Parameter Selection...")

    # 实例化新的 Top-K 选择器
    selector = RobustTopKParamSelector(bo.model, param_space, default_param=[0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0])

    # 提取每个 litmus 前 10 名
    recommendations = selector.select_top_k_params(
        litmus_list=target_litmus_names,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=2.0,  # 惩罚系数
        k=10  # 提取前 10 名
    )

    # 8. 统计与保存结果
    optimized_count = sum(1 for v in recommendations.values() if "optimized" in v['decision'])
    default_count = len(recommendations) - optimized_count

    logger.info(f"Selection Finished.")
    logger.info(f"  - Optimized entries: {optimized_count}")
    logger.info(f"  - Default fallback entries: {default_count}")

    output_file = "best_params_recommendation_robust_topk.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_file}")
    logger.info("=== Done ===")