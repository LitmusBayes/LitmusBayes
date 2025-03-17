import json
import logging
import os
import random
import time
from collections import defaultdict
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
            n_jobs=-1,  # 利用多核
            min_samples_leaf=3,
            random_state=SEED
        )
        # self.model = XGBRegressor(
        #     n_estimators=n_estimators,
        #     learning_rate=0.05,  # 学习率越低越稳，但需要更多 estimator
        #     max_depth=6,  # 树深
        #     subsample=0.8,  # 样本采样
        #     colsample_bytree=0.8,  # 特征采样
        #     n_jobs=-1,
        #     random_state=SEED
        # )
        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)  # 获取 logger

        # 加载向量
        self.litmus_to_vector_dict = {}
        for vec_path in litmus_vec_path:
            litmus_vec_dict = self.load_litmus_vectors(vec_path)
            for vec in litmus_vec_dict:
                self.litmus_to_vector_dict[vec] = litmus_vec_dict[vec]
        # self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

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
        # ============ 关键修改 ============
        # 使用 log1p (log(x+1)) 防止 x=0 报错，同时压缩数值
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    def predict_one(self, litmus_name, param_vec):
        if litmus_name not in self.litmus_to_vector_dict:
            return None
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)
        pred_log = self.model.predict([feature])[0]
        # ============ 关键修改 ============
        # 还原回去：exp(x) - 1
        return np.expm1(pred_log)

    def predict_batch(self, litmus_list, param_list):
        """
        批量预测方法
        :param litmus_list: list of litmus names
        :param param_list: list of param vectors
        :return: predictions array, valid_indices (因为有些litmus可能没有vector)
        """
        X_batch = []
        valid_indices = []

        # 1. 构建特征矩阵
        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]
                X_batch.append(list(param) + list(litmus_vec))
                valid_indices.append(i)

        if not X_batch:
            return [], []

        # 2. 批量预测 (一次性调用，速度极大提升)
        # 注意: np.array(X_batch) 依然有开销，但比循环调用 predict 快得多
        X_batch_np = np.array(X_batch)
        pred_log = self.model.predict(X_batch_np)

        # 3. 还原对数
        preds = np.expm1(pred_log)

        return preds, valid_indices

# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
# litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
litmus_vec_path = "../RQ3/vector/litmus_vector_two_tower_fixed.log"

litmus_vec_path_new = litmus_vec_path
litmus_path_new = litmus_path
cache_file_path = "../RQ1/cache_norm_final.jsonl"
new_chip_data_path = "log_record_init_banana_important_norm_final.log.validation_cache.jsonl"


# ================= 修正后的核心选择类 =================

class RobustParamSelector:
    def __init__(self, model, param_space, default_param=None):
        """
        :param model: 训练好的 sklearn RandomForestRegressor (注意：假设模型是在 log1p 空间训练的)
        :param param_space: LitmusParamSpace 实例
        :param default_param: 兜底参数，默认 [0,2,0,0,0,0,2,0,0,0,0]
        """
        self.model = model
        self.ps = param_space
        self.default_param = default_param if default_param else [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def get_forest_uncertainty(self, X):
        """
        获取随机森林中每棵树的预测，计算 Mean 和 Std
        注意：因为训练时用了 log1p，这里返回的也是 log 空间的 mean 和 std
        """
        # estimators_ 是 sklearn 随机森林存储所有决策树的列表
        # 这一步计算量较大，对于大量候选集可能较慢
        per_tree_pred = [tree.predict(X) for tree in self.model.estimators_]
        per_tree_pred = np.stack(per_tree_pred)  # shape: (n_trees, n_samples)

        means = np.mean(per_tree_pred, axis=0)
        stds = np.std(per_tree_pred, axis=0)

        return means, stds

    def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0):
        """
        为列表中的每个 Litmus test 选择最佳参数
        策略: Score_Log = Mean_Log - (alpha * Std_Log)
        """
        recommendations = {}

        # 1. 获取全量参数空间 (约 20k 个)
        # 修正方法名: get_all_valid_vectors -> get_all_combinations
        all_param_vectors = self.ps.get_all_combinations()
        all_param_vectors = np.array(all_param_vectors)
        n_params = len(all_param_vectors)

        print(f"Searching best params from {n_params} combinations per litmus (Alpha={alpha})...")

        # 为了避免 tqdm 刷屏，我们只在总体循环上做进度条
        for litmus in tqdm(litmus_list, desc="Robust Selection"):
            if litmus not in litmus_feature_map:
                continue

            l_feat = np.array(litmus_feature_map[litmus])

            # 2. 构造输入矩阵: (N_Params, Feat_Dim + Param_Dim)
            # 广播 Litmus 特征
            l_feat_repeated = np.tile(l_feat, (n_params, 1))
            X_batch = np.hstack([all_param_vectors, l_feat_repeated])  # 注意拼接顺序：Param在前还是Feature在前？
            # 检查 RandomForestBO 中的 add 方法: X.append(list(param_vec) + list(litmus_vec))
            # 所以 Param 在前，Feature 在后。上面的代码顺序是正确的。

            # 3. 获取 Log 空间的均值和方差
            # 这里的 means 和 stds 都是 log(y+1) 尺度
            means_log, stds_log = self.get_forest_uncertainty(X_batch)

            # 4. 计算稳健分数 (LCB 策略)
            # 我们在 Log 空间做减法是合理的 (惩罚不确定性)
            robust_scores_log = means_log - (alpha * stds_log)

            # 5. 找到最佳索引
            best_idx = np.argmax(robust_scores_log)

            best_mean_log = means_log[best_idx]
            best_std_log = stds_log[best_idx]
            best_vec = all_param_vectors[best_idx]

            # 6. 还原数值并进行阈值判断
            # 因为模型预测的是 log1p，所以要用 expm1 还原真实分数
            predicted_real_score = np.expm1(best_mean_log)

            final_vec = []
            decision_type = ""

            # 阈值判断：如果预测的真实分数 <= 1.0 (没有加速)，则回退到默认
            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = "optimized"
            else:
                final_vec = self.default_param
                decision_type = "default (low_score)"

            recommendations[litmus] = {
                "param": final_vec,
                "pred_score": float(predicted_real_score),  # 记录还原后的真实分数
                "pred_std_log": float(best_std_log),  # 记录 Log 空间的方差供参考
                "decision": decision_type
            }

        return recommendations


# class PosteriorRFBO:
#     def __init__(self, prior_bo: RandomForestBO, n_estimators=100):
#         """
#         后验模型：利用先验模型的输出作为特征来辅助小样本学习
#         """
#         self.prior_bo = prior_bo
#         # 针对 400 条小样本数据，增加正则化防止过拟合
#         self.model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=6,  # 限制树深
#             min_samples_leaf=5,  # 增加叶节点最小样本数，增强鲁棒性
#             n_jobs=-1,
#             random_state=SEED
#         )
#         self.X_post = []
#         self.y_post = []
#         self.logger = get_logger(LOG_NAME)
#
#     def add_new_data(self, litmus_name, param_vec, score):
#         if litmus_name not in self.prior_bo.litmus_to_vector_dict:
#             return
#
#         litmus_vec = self.prior_bo.litmus_to_vector_dict[litmus_name]
#         base_feature = list(param_vec) + list(litmus_vec)
#
#         # 1. 获取先验模型在 Log 空间的预测值
#         prior_pred_log = self.prior_bo.model.predict([base_feature])[0]
#
#         # 2. 拼接为后验特征：[原始参数, Litmus特征, 先验预测值]
#         post_feature = base_feature + [prior_pred_log]
#
#         self.X_post.append(post_feature)
#         self.y_post.append(score)
#
#     def fit(self):
#         self.logger.info(f"Start fitting Posterior Model with {len(self.X_post)} samples...")
#         y_train_log = np.log1p(np.array(self.y_post))
#         self.model.fit(np.array(self.X_post), y_train_log)
#
#
# class RobustPosteriorSelector(RobustParamSelector):
#     def __init__(self, posterior_bo: PosteriorRFBO, param_space, default_param=None):
#         """
#         继承并覆盖原始的 Selector，使其能够处理带有先验特征的输入
#         """
#         super().__init__(posterior_bo.model, param_space, default_param)
#         self.post_bo = posterior_bo
#         self.prior_bo = posterior_bo.prior_bo
#
#     def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0):
#         recommendations = {}
#         all_param_vectors = np.array(self.ps.get_all_combinations())
#         n_params = len(all_param_vectors)
#
#         print(f"Searching best params using Posterior Model (Alpha={alpha})...")
#
#         for litmus in tqdm(litmus_list, desc="Robust Posterior Selection"):
#             if litmus not in litmus_feature_map:
#                 continue
#
#             l_feat = np.array(litmus_feature_map[litmus])
#             l_feat_repeated = np.tile(l_feat, (n_params, 1))
#
#             # 1. 构造先验模型的基础特征
#             X_base = np.hstack([all_param_vectors, l_feat_repeated])
#
#             # 2. 批量获取先验模型的预测值
#             prior_preds_log = self.prior_bo.model.predict(X_base)
#
#             # 3. 构造后验模型所需的特征矩阵 (在最后增加一列 prior_preds_log)
#             X_post_batch = np.hstack([X_base, prior_preds_log.reshape(-1, 1)])
#
#             # 4. 获取后验模型的均值和方差
#             means_log, stds_log = self.get_forest_uncertainty(X_post_batch)
#
#             # 5. 计算稳健分数并还原
#             robust_scores_log = means_log - (alpha * stds_log)
#             best_idx = np.argmax(robust_scores_log)
#
#             best_mean_log = means_log[best_idx]
#             best_std_log = stds_log[best_idx]
#             best_vec = all_param_vectors[best_idx]
#
#             predicted_real_score = np.expm1(best_mean_log)
#
#             if predicted_real_score > 1.0:
#                 final_vec = best_vec.tolist()
#                 decision_type = "optimized_by_posterior"
#             else:
#                 final_vec = self.default_param
#                 decision_type = "default (low_score)"
#
#             recommendations[litmus] = {
#                 "param": final_vec,
#                 "pred_score": float(predicted_real_score),
#                 "pred_std_log": float(best_std_log),
#                 "decision": decision_type
#             }
#
#         return recommendations
#
class PosteriorRFBO:
    def __init__(self, prior_bo: RandomForestBO, n_estimators=100):
        """
        后验模型：专门用于学习新芯片相对于旧芯片的“残差 (Residual)”
        """
        self.prior_bo = prior_bo
        # 残差模型的超参数可以更保守，因为目标只是微调
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=1,  # 保证叶子节点的泛化性
            n_jobs=-1,
            random_state=SEED
        )
        self.X_post = []
        self.y_residual = []  # 重点：现在我们存的是残差
        self.logger = get_logger(LOG_NAME)

    def add_new_data(self, litmus_name, param_vec, score):
        if litmus_name not in self.prior_bo.litmus_to_vector_dict:
            return

        litmus_vec = self.prior_bo.litmus_to_vector_dict[litmus_name]
        # 现在的特征不需要拼上先验预测了，还是保持原来的 20 维物理特征
        base_feature = list(param_vec) + list(litmus_vec)

        # 1. 获取先验模型在 Log 空间的预测值
        prior_pred_log = self.prior_bo.model.predict([base_feature])[0]

        # 2. 计算残差：真实的 Log 得分 - 先验 Log 预测值
        true_score_log = np.log1p(score)
        residual = true_score_log - prior_pred_log

        self.X_post.append(base_feature)
        self.y_residual.append(residual)

    def fit(self):
        self.logger.info(f"Start fitting Residual Model with {len(self.X_post)} samples...")
        # 直接拟合残差
        self.model.fit(np.array(self.X_post), np.array(self.y_residual))


class RobustPosteriorSelector(RobustParamSelector):
    def __init__(self, posterior_bo: PosteriorRFBO, param_space, default_param=None):
        # 注意：这里传给父类的是残差模型
        super().__init__(posterior_bo.model, param_space, default_param)
        self.post_bo = posterior_bo
        self.prior_bo = posterior_bo.prior_bo

    def select_best_params(self, litmus_list, litmus_feature_map, alpha=2.0):
        recommendations = {}
        all_param_vectors = np.array(self.ps.get_all_combinations_for_best())
        n_params = len(all_param_vectors)

        print(f"Searching best params using Residual Model (Alpha={alpha})...")

        for litmus in tqdm(litmus_list, desc="Robust Residual Selection"):
            if litmus not in litmus_feature_map:
                continue

            l_feat = np.array(litmus_feature_map[litmus])
            l_feat_repeated = np.tile(l_feat, (n_params, 1))

            # 1. 构造基础特征 (20维)
            X_base = np.hstack([all_param_vectors, l_feat_repeated])

            # 2. 获取先验预测 (底座)
            prior_preds_log = self.prior_bo.model.predict(X_base)

            # 3. 获取残差模型的均值和方差 (微调量和不确定性)
            # 这里的 means_res 是残差均值，stds_res 是残差树的方差
            means_res_log, stds_res_log = self.get_forest_uncertainty(X_base)

            # 4. 合并计算最终的 Log 得分
            # 最终期望 = 先验基准 + 残差修正
            # means_res_log *= 2
            final_means_log = prior_preds_log + means_res_log

            # LCB 惩罚：因为先验是确定性输出，所以不确定性全部来源于残差模型
            # robust_scores_log = final_means_log - (alpha * stds_res_log)
            robust_scores_log = final_means_log

            best_idx = np.argmax(robust_scores_log)

            best_mean_log = final_means_log[best_idx]
            best_std_log = stds_res_log[best_idx]
            best_vec = all_param_vectors[best_idx]

            # 5. 还原真实分数
            predicted_real_score = np.expm1(best_mean_log)

            if predicted_real_score > 1.0:
                final_vec = best_vec.tolist()
                decision_type = "optimized_by_residual"
            else:
                final_vec = self.default_param
                decision_type = "default (low_score)"

            recommendations[litmus] = {
                "param": final_vec,
                "pred_score": float(predicted_real_score),
                "pred_std_log": float(best_std_log),  # 这里记录的是残差的不确定性
                "decision": decision_type
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

    # new_litmus_list = get_files(litmus_path_new, '.litmus')
    # litmus_names_new = [path.split("/")[-1][:-7] for path in new_litmus_list]
    litmus_names_new = []
    filter_path = "./a_important_litmus.txt"
    with open(filter_path, "r") as f:
        filter_litmus_list = []
        lines = f.readlines()
        for line in lines:
            litmus_names_new.append(line.strip().split("|")[0].strip())

    # 3. 初始化 BO 和 参数空间
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=200,  # 稍微降低一点树的数量以加快 Robust 搜索速度，或者保持 200
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

    # 7. 执行稳健参数推荐 (Robust Recommendation)
    # ================= 新增：后验模型训练流程 =================
    logger.info("=" * 60)
    logger.info("Building and fitting Posterior Model on NEW chip data...")

    # 假设你的新芯片数据路径如下

    posterior_bo = PosteriorRFBO(prior_bo=bo, n_estimators=50)

    new_data_count = 0
    if os.path.exists(new_chip_data_path):
        with open(new_chip_data_path, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    # 喂给后验模型
                    posterior_bo.add_new_data(obj["litmus"], obj["param"], obj["score"])
                    new_data_count += 1

    if new_data_count > 0:
        posterior_bo.fit()
        logger.info(f"Posterior Model training finished. ({new_data_count} samples)")
    else:
        logger.error("No new chip data found! Exiting.")
        exit(1)

    # ================= 修改：使用后验选择器 =================
    logger.info("=" * 60)
    logger.info("Starting Robust Parameter Selection with Posterior Model...")

    # 实例化新的后验选择器
    selector = RobustPosteriorSelector(
        posterior_bo=posterior_bo,
        param_space=param_space,
        default_param=[0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    )

    # 这里的调用与之前相同，但内部逻辑已经升级为后验融合
    recommendations = selector.select_best_params(
        litmus_list=litmus_names_new,
        litmus_feature_map=bo.litmus_to_vector_dict,
        alpha=2.0  # 针对新芯片如果把握不大，可以将 alpha 调高到 1.5 增加保守性
    )

    # 8. 统计与保存结果
    optimized_count = sum(1 for v in recommendations.values() if v['decision'] == 'optimized')
    default_count = len(recommendations) - optimized_count

    logger.info(f"Selection Finished.")
    logger.info(f"  - Optimized: {optimized_count}")
    logger.info(f"  - Default (Safety Fallback): {default_count}")

    output_file = "best_params_recommendation_robust_final.json"
    with open(output_file, "w") as f:
        json.dump(recommendations, f, indent=4)

    logger.info(f"Recommendations saved to: {output_file}")
    logger.info("=== Done ===")