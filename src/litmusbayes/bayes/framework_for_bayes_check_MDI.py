import json
import logging
import os
import random
import time
from collections import defaultdict
from xgboost import XGBRegressor
from scipy.stats import norm, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
import seaborn as sns
# 引入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np

SEED = 2025
LOG_NAME = "bayes_eval"


def visualize_model_analysis(bo, test_data, top_n=15):
    """
    生成三张图：
    1. MDI Feature Importance
    2. Permutation Feature Importance
    3. Prediction vs Actual Scatter Plot
    """
    # 设置风格
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 尝试支持中文，如果乱码请忽略
    plt.rcParams['axes.unicode_minus'] = False

    print("准备绘图数据...")

    # --- 1. 准备数据 ---
    # 提取测试集特征和标签
    X_test = []
    y_test_raw = []  # 真实值
    y_pred_raw = []  # 预测值 (反变换后的)

    # 辅助：获取特征名称
    # 假设第一条数据存在，用来推断维度
    sample_item = test_data[0]
    n_params = len(sample_item['param'])
    n_litmus = len(bo.litmus_to_vector_dict[sample_item['litmus']])

    feature_names = [f"Param_{i}" for i in range(n_params)] + \
                    [f"LitmusVec_{i}" for i in range(n_litmus)]

    for item in test_data:
        if item['litmus'] in bo.litmus_to_vector_dict:
            l_vec = bo.litmus_to_vector_dict[item['litmus']]
            feat = list(item['param']) + list(l_vec)
            X_test.append(feat)
            y_test_raw.append(item['score'])
            # 预测 (注意：predict_one 内部已经做了 expm1)
            pred = bo.predict_one(item['litmus'], item['param'])
            y_pred_raw.append(pred)

    X_test_np = np.array(X_test)
    y_test_log = np.log1p(np.array(y_test_raw))  # 用于 Permutation Importance 计算 (需与训练目标一致)

    # 创建画布：2行2列 (最后一个位置留空或画别的)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)

    # --- 图 1: MDI Feature Importance (模型内部权重) ---
    ax1 = fig.add_subplot(gs[0, 0])
    importances = bo.model.feature_importances_
    # 排序
    indices = np.argsort(importances)[::-1][:top_n]

    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax1, palette="viridis")
    ax1.set_title("MDI Feature Importance (Train Set)", fontsize=14)
    ax1.set_xlabel("Importance Score")

    # --- 图 2: Permutation Importance (测试集验证) ---
    print("计算 Permutation Importance (这可能需要几秒钟)...")
    ax2 = fig.add_subplot(gs[0, 1])
    result = permutation_importance(
        bo.model, X_test_np, y_test_log,
        n_repeats=5, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error'
    )
    perm_sorted_idx = result.importances_mean.argsort()[::-1][:top_n]

    sns.barplot(x=result.importances_mean[perm_sorted_idx],
                y=[feature_names[i] for i in perm_sorted_idx], ax=ax2, palette="magma")
    ax2.set_title("Permutation Importance (Test Set)", fontsize=14)
    ax2.set_xlabel("Decrease in Accuracy (MSE)")

    # --- 图 3: Predicted vs Actual Scatter Plot (关键诊断) ---
    ax3 = fig.add_subplot(gs[1, :])  # 占满第二行

    # 计算指标
    r2 = r2_score(y_test_raw, y_pred_raw)
    rho, _ = spearmanr(y_test_raw, y_pred_raw)
    pearson, _ = pearsonr(y_test_raw, y_pred_raw)

    # 绘制散点
    sns.scatterplot(x=y_test_raw, y=y_pred_raw, alpha=0.5, ax=ax3, color='royalblue', label='Samples')

    # 绘制理想线 y=x
    min_val = min(min(y_test_raw), min(y_pred_raw))
    max_val = max(max(y_test_raw), max(y_pred_raw))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')

    ax3.set_title(f"Predicted vs Actual Score\nRho={rho:.4f} | R2={r2:.4f} | Pearson={pearson:.4f}", fontsize=16)
    ax3.set_xlabel("Actual Score (True)", fontsize=12)
    ax3.set_ylabel("Predicted Score (Model)", fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = "model_diagnosis.png"
    plt.savefig(save_path, dpi=100)
    print(f"\n[完成] 诊断图已保存至: {save_path}")
    print("请查看该图片进行分析。")
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
                 litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,  # 利用多核
            max_features="sqrt",
            min_samples_leaf=10,
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
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

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
            print("error",litmus_name)
            assert False
            # return
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


# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"
cache_file_path = stat_log_base + ".cache_sum_70_no.jsonl"

if __name__ == "__main__":
    # 1. Setup Logger
    random.seed(SEED)
    np.random.seed(SEED)

    # 日志文件命名
    log_file_name = f"{stat_log_base}.check.run.log"
    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Evaluation Run | Seed={SEED} ===")

    # 2. 读取 Litmus List
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=200,
        litmus_vec_path=litmus_vec_path
    )

    # 4. 加载 Cache
    logger.info(f"Loading data from {cache_file_path} ...")
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass
    else:
        logger.error("Cache file not found!")
        exit(1)

    total_count = len(all_data)
    logger.info(f"Total records loaded: {total_count}")

    if total_count <= 15000:
        logger.warning("Data size <= 7000, splitting might be invalid based on request.")

    # 5. 切分数据
    train_data = all_data[:50000]
    test_data = all_data[50000:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 构建训练集 & 训练
    logger.info("Building training set...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])

    bo.fit()

    logger.info("=== Starting Model Diagnosis & Visualization ===")

    # 引入可视化函数 (确保 visualize_model_analysis 已经定义或import)
    try:
        # 这个函数内部会计算 MDI 和 Permutation Importance，并画出图片
        # 还会把关键指标打印在控制台，所以不需要外面那段冗余代码了
        visualize_model_analysis(bo, test_data, top_n=15)

        logger.info("Visualization saved to 'model_diagnosis.png'. Check it for details.")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback

        traceback.print_exc()

    logger.info("=== All Done ===")