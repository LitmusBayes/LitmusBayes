import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
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

SEED = 2025
LOG_NAME = "bayes_eval"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# --- 自定义 Loss: 混合了数值误差和方向误差 ---
class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(PearsonMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # 控制 MSE 和 Cosine 的权重比例

    def forward(self, pred, target):
        # 1. 数值准确性 (MSE)
        loss_mse = self.mse(pred, target)

        # 2. 形状相似性 (Cosine Embedding Loss 的变体)
        # 我们希望 pred 和 target 的向量方向一致 (即 Cosine Similarity = 1)
        # Cosine Loss = 1 - CosineSimilarity
        loss_cos = 1 - F.cosine_similarity(pred, target, dim=1).mean()

        # 组合 Loss
        return (1 - self.alpha) * loss_mse + self.alpha * loss_cos


# --- 增强版模型 ---
class LitmusEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=16):
        super(LitmusEmbedder, self).__init__()

        # 编码器：加深网络，增加非线性能力
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # 加上 BatchNorm 加速收敛
            nn.LeakyReLU(0.2),  # 使用 LeakyReLU 防止神经元坏死

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, embedding_dim)  # 输出 Embedding
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        emb = self.encoder(x)
        # 对 Embedding 做一个归一化，有时候有助于 RSA
        # emb = F.normalize(emb, p=2, dim=1)
        pred = self.decoder(emb)
        return pred, emb

def analyze_embedding_quality(embeddings, performances, method_name="My Embedding"):
    """
    embeddings: (N, D) 你的向量
    performances: (N, 70) 你的运行结果
    """

    # --- 1. 计算 Ground Truth (性能趋势距离) ---
    # metric='correlation' 自动处理归一化，只关注波动的形状
    # pdist 返回的是压缩后的距离向量（只包含上三角）
    dist_perf_vector = pdist(performances, metric='correlation')

    # --- 2. 计算 Embedding Distance (嵌入距离) ---
    # 常用 'cosine' 或 'euclidean'
    dist_emb_vector = pdist(embeddings, metric='cosine')

    # --- 3. 统计检验 (Spearman Correlation) ---
    # 比较两个距离向量的排序一致性
    rho, p_val = spearmanr(dist_emb_vector, dist_perf_vector)

    print(f"=== {method_name} 评估结果 ===")
    print(f"Spearman Correlation (RSA Score): {rho:.4f}")
    print(f"P-value: {p_val:.4e}")
    if p_val < 0.05:
        print("结论: 显著相关 (p < 0.05)")
    else:
        print("结论: 不相关 (统计不显著)")

    # --- 4. 可视化 (可选，但强烈推荐) ---
    # 将距离向量还原回 N*N 矩阵以便绘图
    mat_perf = squareform(dist_perf_vector)
    mat_emb = squareform(dist_emb_vector)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(mat_perf, ax=axes[0], cmap="viridis", cbar=True)
    axes[0].set_title("Ground Truth: Performance Trend Distances")
    axes[0].set_xlabel("Litmus Tests")
    axes[0].set_ylabel("Litmus Tests")

    sns.heatmap(mat_emb, ax=axes[1], cmap="viridis", cbar=True)
    axes[1].set_title(f"Embedding Space Distances\n(Score: {rho:.3f})")
    axes[1].set_xlabel("Litmus Tests")

    plt.tight_layout()
    plt.show()

    return rho

def perf_vector_emb(n_tests, n_params, emb_num, emb, real_perf):

    # ==========================================
    # 模拟你的数据场景进行测试
    # ==========================================
    np.random.seed(42)
    DIM_EMB = 14


    # 机制 B: 这是一个“烂”的嵌入 (完全随机)
    bad_emb = np.random.rand(n_tests, emb_num)

    # 3. 运行评估
    score_a = analyze_embedding_quality(emb, real_perf, "机制 A (Good)")
    score_b = analyze_embedding_quality(bad_emb, real_perf, "机制 B (Random)")

    print(f"\n最终对比: 机制 A ({score_a:.3f}) vs 机制 B ({score_b:.3f})")





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


def train_and_get_embeddings(raw_features, target_performances, embedding_dim=16, epochs=500):
    """
    raw_features: (N, 13)
    target_performances: (N, 70)
    """
    # === 关键步骤 1: 数据 Log 变换 ===
    # 神经网络极其讨厌数量级差异大的数据。
    # 用 log1p 把 [0, 10000] 压缩到 [0, 9] 左右，这会让训练容易 100 倍。

    # 1. 输入特征归一化 (StandardScaler 手写版)
    X_mean = raw_features.mean(axis=0)
    X_std = raw_features.std(axis=0) + 1e-6
    X_norm = (raw_features - X_mean) / X_std
    X = torch.FloatTensor(X_norm)

    # 2. 目标值 Log 变换
    # 如果你的 score 包含 0，log1p 是安全的 (log(x+1))
    target_log = np.log1p(target_performances)
    y = torch.FloatTensor(target_log)

    # 创建 DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # 增大 batch_size

    # 初始化
    input_dim = raw_features.shape[1]
    output_dim = target_performances.shape[1]

    model = LitmusEmbedder(input_dim, output_dim, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.002)  # 稍微调大初始学习率

    # === 关键步骤 2: 使用混合 Loss ===
    # alpha=0.7 表示我们 70% 关注形状(RSA)，30% 关注数值(MSE)
    criterion = PearsonMSELoss(alpha=0.7)

    # 动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    print(f"\n>>> 开始增强训练 (Input:{input_dim} -> Emb:{embedding_dim} -> Out:{output_dim})")
    print(">>> 这里的 Loss 越低，说明预测的'趋势'越准")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds, _ = model(batch_x)

            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)  # 根据 Loss 调整学习率

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.5f}")

    # 提取嵌入
    print(">>> 训练完成，正在提取新嵌入...")
    model.eval()
    with torch.no_grad():
        # 注意：这里输入网络的是归一化后的 X
        _, new_embeddings = model(X)

    return new_embeddings.numpy()
# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm.jsonl"

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

    if total_count <= 10000:
        logger.warning("Data size <= 7000, splitting might be invalid based on request.")

    # 5. 切分数据
    random.shuffle(all_data)
    train_data = all_data[:60000]
    test_data = all_data[60000:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 构建训练集 & 训练
    logger.info("Building training set...")
    param_dict = {}
    param_num = 0
    litmus_dict = {}
    litmus_num = 0
    emd_num = 13
    for item in all_data:
        if str(item["param"]) not in param_dict:
            param_dict[str(item["param"])] = param_num
            param_num += 1
        if item["litmus"] not in litmus_dict:
            litmus_dict[item["litmus"]] = litmus_num
            litmus_num += 1

    print(litmus_num)
    print(param_num)
    performance = np.zeros((litmus_num,param_num))
    emb = np.zeros((litmus_num,emd_num))

    for item in all_data:
        emb[litmus_dict[item["litmus"]]] = bo.litmus_to_vector_dict[f"{item["litmus"]}"]
        performance[litmus_dict[item["litmus"]]][param_dict[str(item["param"])]] = item["score"]

    # perf_vector_emb(litmus_num, param_num, emd_num, emb, performance)
        # 1. 先评估一下原始的“烂”向量 (作为 Baseline)
    print("\n------------------------------------------------")
    print("【基准测试】原始手工特征 (13维) 的质量：")
    analyze_embedding_quality(emb, performance, method_name="Original Features")

    # 2. 训练神经网络，学习新的嵌入
    # 我们设定学习一个 16 维的向量
    learned_emb = train_and_get_embeddings(emb, performance, embedding_dim=16, epochs=300)

    # 3. 评估神经网络学到的向量
    print("\n------------------------------------------------")
    print("【神经网络】学习到的嵌入 (16维) 的质量：")
    analyze_embedding_quality(learned_emb, performance, method_name="Neural Network Embedding")