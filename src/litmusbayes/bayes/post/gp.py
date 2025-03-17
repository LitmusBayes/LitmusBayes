import json
import logging
import os
import random
import time
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

# 你的自定义模块 (假设路径正确)
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "posterior_gen"


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


# ================= 2. 后验增强器 (Posterior Booster) (保持不变) =================

class PosteriorBooster:
    def __init__(self, rf_model, litmus_vec_dict):
        self.rf_model = rf_model
        self.litmus_vec_dict = litmus_vec_dict

        # 定义 GP 核函数：Constant(幅度) * Matern(形状) + White(噪声)
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + \
                 WhiteKernel(noise_level=0.1)

        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=SEED)
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
                        real_log = np.log1p(score)
                        residual = real_log - rf_pred_log

                        self.X_train.append(feature)
                        self.y_residuals.append(residual)
                        count += 1
                    except:
                        pass
        print(f"Loaded {count} residual points from Chip B.")

    def fit(self):
        if not self.X_train:
            print("No Chip B data found. GP will not run.")
            return
        print("Fitting GP on residuals...")
        self.gp.fit(np.array(self.X_train), np.array(self.y_residuals))
        self.is_fitted = True
        print("GP Fitted.")

    def predict_batch_log(self, X_batch, return_std=True):
        """
        批量预测后验 Log 分数
        Posterior_Log = RF_Log(X) + GP_Residual_Log(X)
        """
        # 1. RF Base
        mu_rf = self.rf_model.predict(X_batch)

        # 2. GP Correction
        if self.is_fitted:
            if return_std:
                mu_gp, std_gp = self.gp.predict(X_batch, return_std=True)
            else:
                mu_gp = self.gp.predict(X_batch)
                std_gp = np.zeros_like(mu_gp)
        else:
            mu_gp = np.zeros_like(mu_rf)
            std_gp = np.zeros_like(mu_rf)

        # 3. Combine
        mu_total = mu_rf + mu_gp
        return mu_total, std_gp


# ================= 3. 后验选择器 (修改了 select_all 方法) =================

class PosteriorSelector:
    def __init__(self, booster, param_space):
        self.booster = booster
        self.ps = param_space
        # 默认参数 (用于兜底)
        self.default_param = [0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    def select_all(self, litmus_list, alpha=1.0):
        """
        为每个 litmus test 选择最佳参数。
        利用矩阵运算加速：一次性预测一个 Litmus 的所有参数组合。
        输出详细的 JSON 结构。
        """
        recommendations = {}

        # 1. 获取所有候选参数组合
        all_params = np.array(self.ps.get_all_combinations())
        n_params = len(all_params)

        print(f"Selecting best params for {len(litmus_list)} tests...")
        print(f"Candidate Space Size: {n_params}")

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

            # 3. 后验预测 (RF + GP)
            # mu_log: 预测的 Log 分数
            # std_log: GP 的不确定性
            mu_log, std_log = self.booster.predict_batch_log(X_batch, return_std=True)

            # 4. 决策逻辑 (LCB)
            scores = mu_log - (alpha * std_log)

            # 5. 选最佳索引
            best_idx = np.argmax(scores)

            # 获取该点的详细信息
            best_mu_log = mu_log[best_idx]
            best_std_log = std_log[best_idx]

            # 还原预测分数 (log1p -> expm1)
            predicted_score_linear = np.expm1(best_mu_log)

            # 6. 阈值检查与结构组装
            # 注意：np 类型不能直接被 JSON 序列化，需要转 float()
            result_obj = {
                "param": all_params[best_idx].tolist(),
                "pred_score": float(predicted_score_linear),
                "pred_std_log": float(best_std_log),
                "decision": "optimized"
            }

            # 如果预测的 Log 分数太低 (比如接近 0)，说明优化也没用，退回默认
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
    OUTPUT_JSON = "./posterior_best_params.json"

    # ==========================

    # 修复日志路径问题
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

    # 2. 训练 GP
    logger.info("Step 2: Training Posterior Model (GP) on Chip B...")
    booster = PosteriorBooster(bo.model, bo.litmus_to_vector_dict)
    booster.load_chip_b_data([CHIP_B_LOG])
    booster.fit()

    # 3. 全量推断
    logger.info("Step 3: Selecting Best Parameters for ALL Litmus Tests...")

    all_files = get_files(LITMUS_PATH)
    all_litmus_names = [p.split("/")[-1][:-7] for p in all_files]

    selector = PosteriorSelector(booster, param_space)

    # 这一步现在会返回包含详细 dict 的结构
    best_params_map = selector.select_all(all_litmus_names, alpha=1.0)

    # 4. 保存结果
    logger.info(f"Step 4: Saving {len(best_params_map)} configurations to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(best_params_map, f, indent=4)

    logger.info("Done. You can now use this JSON to run tests on Chip B.")