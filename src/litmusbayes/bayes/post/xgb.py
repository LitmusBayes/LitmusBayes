import json
import logging
import os
import random
import time
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

# === 新增 XGBoost ===
from xgboost import XGBRegressor

# 你的自定义模块 (保持不变)
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "posterior_gen_xgb"


# ================= 1. 基础工具类 (保持不变) =================

class RandomForestBO:
    def __init__(self, param_space, n_estimators=100, litmus_vec_paths=[]):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=-1, min_samples_leaf=3, random_state=SEED
        )
        self.X = []
        self.y = []
        self.litmus_to_vector_dict = {}
        for path in litmus_vec_paths:
            self._load_vectors(path)

    def _load_vectors(self, path):
        if not os.path.exists(path): return
        with open(path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    self.litmus_to_vector_dict[k] = eval(v)

    def add(self, litmus_name, param_vec, score):
        if litmus_name in self.litmus_to_vector_dict:
            vec = self.litmus_to_vector_dict[litmus_name]
            self.X.append(list(param_vec) + list(vec))
            self.y.append(score)

    def fit(self):
        # 训练 RF (Log 空间)
        y_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_log)

    def predict_log(self, X):
        return self.model.predict(X)


# ================= 2. 后验增强器 (Posterior Booster) - 改为 XGBoost =================

class PosteriorBooster:
    def __init__(self, rf_model, litmus_vec_dict):
        self.rf_model = rf_model
        self.litmus_vec_dict = litmus_vec_dict

        # === 修改点：使用 XGBoost ===
        # 树模型不需要特征归一化，且对混合特征处理更好
        self.xgb = XGBRegressor(
            n_estimators=100,  # 树的数量，数据少可适当减少
            max_depth=4,  # 树深，防止过拟合
            learning_rate=0.05,  # 学习率，越小越稳
            subsample=0.8,  # 每次只用80%的数据，增加鲁棒性
            colsample_bytree=0.8,  # 每次只用80%的特征
            n_jobs=-1,
            random_state=SEED
        )

        self.X_train = []
        self.y_residuals = []
        self.is_fitted = False

    def load_chip_b_data(self, log_paths):
        """加载 Chip B 的数据，计算相对于 Chip A RF 的残差"""
        print(f"Loading Chip B data from {log_paths}...")
        count = 0
        for path in log_paths:
            if not os.path.exists(path): continue
            with open(path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        litmus, param, score = obj['litmus'], obj['param'], obj['score']

                        if litmus not in self.litmus_vec_dict: continue

                        # 构造特征
                        l_vec = self.litmus_vec_dict[litmus]
                        feature = list(param) + list(l_vec)

                        # 计算残差: Log(Real_B) - Log(Pred_A)
                        rf_pred_log = self.rf_model.predict([feature])[0]

                        # === 安全性检查 ===
                        # 只有当 score 大于极小值时才计算 Log，防止 Log(0) 报错或数值爆炸
                        if score < 1e-9:
                            continue

                        real_log = np.log1p(score)
                        residual = real_log - rf_pred_log

                        self.X_train.append(feature)
                        self.y_residuals.append(residual)
                        count += 1
                    except Exception as e:
                        # print(e)
                        pass
        print(f"Loaded {count} residual points from Chip B.")

    def fit(self):
        if not self.X_train:
            print("No Chip B data found. XGBoost will not run.")
            return

        print(f"Fitting XGBoost on {len(self.X_train)} residuals...")

        # XGBoost 可以直接吃 List[List]，也可以吃 numpy array
        # 树模型不需要 StandardScaler
        self.xgb.fit(np.array(self.X_train), np.array(self.y_residuals))
        self.is_fitted = True
        print("XGBoost Fitted.")

    def predict_batch_log(self, X_batch, return_std=True):
        """
        批量预测后验 Log 分数
        Posterior_Log = RF_Log(X) + XGB_Residual_Log(X)
        """
        # 1. RF Base
        mu_rf = self.rf_model.predict(X_batch)

        # 2. XGB Correction
        if self.is_fitted:
            mu_xgb = self.xgb.predict(X_batch)
        else:
            mu_xgb = np.zeros_like(mu_rf)

        # === 注意 ===
        # 标准 XGBoost 不输出 std (不确定性)。
        # 为了兼容你的接口，这里我们返回 0 向量。
        # 这意味着后续的 Selector 会变成纯贪婪模式 (只看均值)。
        std_xgb = np.zeros_like(mu_rf)

        # 3. Combine
        mu_total = mu_rf + mu_xgb

        return mu_total, std_xgb


# ================= 3. 后验选择器 (保持不变，逻辑兼容) =================

class PosteriorSelector:
    def __init__(self, booster, param_space):
        self.booster = booster
        self.ps = param_space
        # 默认参数 (用于兜底)
        self.default_param = [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def select_all(self, litmus_list, alpha=1.0):
        """
        为每个 litmus test 选择最佳参数。
        """
        recommendations = {}

        # 1. 获取所有候选参数组合
        all_params = np.array(self.ps.get_all_combinations())
        n_params = len(all_params)

        print(f"Selecting best params for {len(litmus_list)} tests...")

        for litmus in tqdm(litmus_list, desc="Posterior Inference"):
            # 情况 A: 缺少向量信息 -> 只能给默认值
            if litmus not in self.booster.litmus_vec_dict:
                recommendations[litmus] = {
                    "param": self.default_param,
                    "pred_score": 0.0,
                    "pred_std_log": 0.0,
                    "decision": "default_no_vector"
                }
                continue

            # 2. 构造输入矩阵 X_batch: (N_Params, Dim)
            l_vec = np.array(self.booster.litmus_vec_dict[litmus])
            l_vec_repeated = np.tile(l_vec, (n_params, 1))
            X_batch = np.hstack([all_params, l_vec_repeated])

            # 3. 后验预测 (RF + XGB)
            # mu_log: 预测的 Log 分数
            # std_log: 这里现在全是 0
            mu_log, std_log = self.booster.predict_batch_log(X_batch, return_std=True)

            # 4. 决策逻辑
            # 因为 std_log 是 0，所以 alpha 不起作用了，这里变成纯粹选预测分最高的
            scores = mu_log

            # 5. 选最佳索引
            best_idx = np.argmax(scores)

            best_mu_log = mu_log[best_idx]
            best_std_log = std_log[best_idx]  # 0.0

            predicted_score_linear = np.expm1(best_mu_log)

            result_obj = {
                "param": all_params[best_idx].tolist(),
                "pred_score": float(predicted_score_linear),
                "pred_std_log": float(best_std_log),
                "decision": "optimized_xgb"
            }

            if best_mu_log <= 0.05:
                result_obj["param"] = self.default_param
                result_obj["decision"] = "default_low_score"

            recommendations[litmus] = result_obj

        return recommendations


# ================= 4. 主程序 =================

if __name__ == "__main__":
    # === 配置路径 ===
    # Chip A (历史数据)
    CHIP_A_CACHE = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache4_norm.jsonl"
    LITMUS_VEC = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"

    # Chip B (新芯片数据 - 你的聚类运行日志)
    CHIP_B_LOG = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_representative_norm_run.log.cache.jsonl"

    # 目标 Litmus 列表
    LITMUS_PATH = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"

    # 输出文件
    OUTPUT_JSON = "./posterior_best_params_xgb.json"

    # ==========================

    logger = setup_logger(f"./{LOG_NAME}.log", logging.INFO, LOG_NAME, True)
    logger.info("Step 1: Training Prior Model (Random Forest) on Chip A...")

    param_space = LitmusParamSpace()

    # 1. 训练 RF
    bo = RandomForestBO(param_space, litmus_vec_paths=[LITMUS_VEC])

    if os.path.exists(CHIP_A_CACHE):
        with open(CHIP_A_CACHE, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    bo.add(obj["litmus"], obj["param"], obj["score"])
                except:
                    pass
    else:
        logger.error("Chip A cache not found!")
        exit(1)

    bo.fit()
    logger.info("Prior Model Trained.")

    # 2. 训练 XGBoost
    logger.info("Step 2: Training Posterior Model (XGBoost) on Chip B...")
    booster = PosteriorBooster(bo.model, bo.litmus_to_vector_dict)
    booster.load_chip_b_data([CHIP_B_LOG])
    booster.fit()

    # 3. 全量推断
    logger.info("Step 3: Selecting Best Parameters for ALL Litmus Tests...")

    all_files = get_files(LITMUS_PATH)
    all_litmus_names = [p.split("/")[-1][:-7] for p in all_files]

    selector = PosteriorSelector(booster, param_space)

    # 这里的 alpha 随便填，因为 std 为 0
    best_params_map = selector.select_all(all_litmus_names, alpha=1.0)

    # 4. 保存结果
    logger.info(f"Step 4: Saving {len(best_params_map)} configurations to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(best_params_map, f, indent=4)

    logger.info("Done.")