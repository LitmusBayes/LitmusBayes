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


# ================= 新增：多样性参数选择器 =================

class DiverseParamSelector:
    def __init__(self, model, param_space):
        """
        :param model: 训练好的 sklearn RandomForestRegressor
        :param param_space: LitmusParamSpace 实例
        """
        self.model = model
        self.ps = param_space

    def get_forest_uncertainty(self, X):
        """
        获取随机森林中每棵树的预测，计算 Mean 和 Std (Log 空间)
        """
        per_tree_pred = [tree.predict(X) for tree in self.model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)  # shape: (n_trees, n_samples)

        means = np.mean(per_tree_pred, axis=0)
        stds = np.std(per_tree_pred, axis=0)
        return means, stds

    def select_diverse_params(self, target_litmus_list, litmus_feature_map, k=5, pool_ratio=0.1):
        """
        为目标 Litmus 列表生成均匀分布的高分参数
        """
        final_recommendations = {}

        # 1. 获取全量参数空间
        all_param_vectors = self.ps.get_all_combinations()
        X_base_params = np.array(all_param_vectors)
        num_total_params = len(X_base_params)

        # 确定候选池大小
        pool_size = max(k, int(num_total_params * pool_ratio))

        print(f"Generating {k} diverse params for each of {len(target_litmus_list)} litmus tests...")
        print(f"Sampling from the top {pool_size} (pool_ratio={pool_ratio}) predictions.")

        for litmus in tqdm(target_litmus_list, desc="Diverse Selection"):
            if litmus not in litmus_feature_map:
                continue

            l_feat = np.array(litmus_feature_map[litmus])

            # 2. 构造批量预测特征矩阵: (N_Params, Param_Dim + Feat_Dim)
            l_feat_repeated = np.tile(l_feat, (num_total_params, 1))
            X_batch = np.hstack([X_base_params, l_feat_repeated])

            # 3. 获取 Log 空间的均值和方差
            means_log, stds_log = self.get_forest_uncertainty(X_batch)

            # 将均值还原为真实分数

            preds_real = np.expm1(means_log)

            # 4. 筛选前 pool_size 个高分参数 (基于均值)
            top_indices = np.argsort(preds_real)[-pool_size:]
            top_params = X_base_params[top_indices]
            top_preds_real = preds_real[top_indices]
            top_stds_log = stds_log[top_indices]

            # 5. K-Means 聚类以保证空间分布
            # n_init="auto" 是 sklearn 最新版本的推荐写法，跑得更快
            kmeans = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
            cluster_labels = kmeans.fit_predict(top_params)

            # 6. 在每个簇中选取得分最高的参数
            for cluster_id in range(k):
                in_cluster_idx = np.where(cluster_labels == cluster_id)[0]
                if len(in_cluster_idx) == 0:
                    continue

                # 找到该簇内 pred_score 最高的索引
                best_in_cluster_idx = in_cluster_idx[np.argmax(top_preds_real[in_cluster_idx])]

                selected_param = top_params[best_in_cluster_idx].tolist()
                selected_score = top_preds_real[best_in_cluster_idx]
                selected_std = top_stds_log[best_in_cluster_idx]

                # 构造符合要求的 Key，例如 "SB+rfi-ctrlfencei+pos-rfi-addr_cluster_0"
                result_key = f"{litmus}_cluster_{cluster_id}"

                final_recommendations[result_key] = {
                    "param": selected_param,
                    "pred_score": float(selected_score),
                    "pred_std_log": float(selected_std),
                    "decision": f"optimized_cluster_{cluster_id}"
                }

        return final_recommendations


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
    logger.info(f"=== Start Diverse Parameter Selection | Seed={SEED} ===")

    # （如果目标文件不存在，自动生成一个假的测试文件防止报错）
    if not os.path.exists(target_litmus_txt):
        logger.warning(f"Target list {target_litmus_txt} not found. Creating a dummy one for testing.")
        with open(target_litmus_txt, "w") as f:
            f.write("SB+pos-rfi-addr+pos-rfi-ctrlfencei\n")
            f.write("SB+pos-po-addrs\n")
            f.write("SB+po-addr+pos-po-addr\n")

    # 2. 读取要采样的目标 Litmus 文件列表
    with open(target_litmus_txt, 'r') as f:
        target_litmus_names = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(target_litmus_names)} target litmus tests to sample.")

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

    # 6. 执行多样性参数选择 (K-Means 采样)
    logger.info("=" * 60)
    logger.info("Starting Diverse Parameter Selection...")

    selector = DiverseParamSelector(bo.model, param_space)

    # 你可以修改 k (每个 litmus 选多少个) 和 pool_ratio (在前百分之多少里找)
    recommendations = selector.select_diverse_params(
        target_litmus_list=target_litmus_names,
        litmus_feature_map=bo.litmus_to_vector_dict,
        k=20,  # <--- 根据你的要求，这里可以改成你想要的任意数量，比如 2 或者 5
        pool_ratio=0.005  # <--- 在排名前 10% 的高分参数中进行空间均匀采样
    )

    # 7. 保存结果
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Successfully generated {len(recommendations)} parameter configs.")
    logger.info(f"Recommendations saved to: {output_json_path}")
    logger.info("=== Done ===")