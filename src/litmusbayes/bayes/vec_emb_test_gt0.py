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

from sklearn.model_selection import train_test_split
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


def run_train_and_inference(X_train, y_train, X_test, embedding_dim=16, epochs=500):
    """
    X_train: (N_train, 13) - 手工特征
    y_train: (N_train, 70) - 性能数据
    X_test:  (N_test, 13)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. 数据归一化 (防止泄漏) ===
    # 仅使用训练集的均值和方差
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6  # 防止除零

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # === 2. Log 变换 Performance ===
    y_train_log = np.log1p(y_train)

    # === 3. 转 Tensor ===
    xt_tensor = torch.FloatTensor(X_train_norm)
    yt_tensor = torch.FloatTensor(y_train_log)
    xv_tensor = torch.FloatTensor(X_test_norm)

    dataset = TensorDataset(xt_tensor, yt_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # === 4. 初始化模型 ===
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = LitmusEmbedder(input_dim, output_dim, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = PearsonMSELoss(alpha=0.7).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    print(f"\n>>> Start Training (Input:{input_dim} -> Emb:{embedding_dim})...")

    # === 5. 训练 ===
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.5f}")

    # === 6. 推理 ===
    print(">>> Generating embeddings...")
    model.eval()

    with torch.no_grad():
        _, emb_train = model(xt_tensor.to(device))
        emb_train = emb_train.cpu().numpy()

        _, emb_test = model(xv_tensor.to(device))
        emb_test = emb_test.cpu().numpy()

    return emb_train, emb_test

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_gt_0_for_graph.jsonl"

# ==========================================
# 1. 辅助类 (用于加载 13维 向量)
# ==========================================
class VectorLoader:
    """
    简化版的加载器，只负责读取 litmus_vector.log
    """
    def __init__(self, vector_path):
        self.litmus_to_vec = {}
        if os.path.exists(vector_path):
            with open(vector_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line: continue
                    name, vec_str = line.split(":", 1)
                    try:
                        vec = eval(vec_str)
                        self.litmus_to_vec[name] = list(vec)
                    except:
                        pass
        else:
            print(f"Error: Vector file not found at {vector_path}")

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger(f"{stat_log_base}.check.run.log", level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Manual-Feature Evaluation (Split) | Seed={SEED} ===")

    # 1. 加载 Performance 数据
    logger.info("Loading performance data...")
    all_raw_data = []
    unique_names = set()

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_raw_data.append(item)
                    unique_names.add(item["litmus"])
    else:
        logger.error("Cache file not found!")
        exit(1)

    all_litmus_list = sorted(list(unique_names))

    # 2. 【核心修改】划分数据集 (按名字)
    train_names, test_names = train_test_split(all_litmus_list, test_size=0.20, random_state=SEED)
    logger.info(f"Total Programs: {len(all_litmus_list)}")
    logger.info(f"Train Set: {len(train_names)} | Test Set: {len(test_names)}")

    # 3. 加载 13维 手工向量
    vec_loader = VectorLoader(litmus_vec_path)
    logger.info(f"Loaded {len(vec_loader.litmus_to_vec)} vectors from file.")

    # 4. 构建全量矩阵
    # 统一参数列索引
    all_params = sorted(list(set([str(item["param"]) for item in all_raw_data])))
    param_dict = {p: i for i, p in enumerate(all_params)}
    litmus_dict = {name: i for i, name in enumerate(all_litmus_list)}

    full_perf = np.zeros((len(all_litmus_list), len(all_params)))
    full_feat = np.zeros((len(all_litmus_list), 13))  # 假设手工向量是13维

    # 填充 Performance 矩阵
    for item in all_raw_data:
        l_idx = litmus_dict[item["litmus"]]
        p_idx = param_dict[str(item["param"])]
        full_perf[l_idx][p_idx] = item["score"]

    # 填充 Feature 矩阵
    for name, idx in litmus_dict.items():
        if name in vec_loader.litmus_to_vec:
            full_feat[idx] = np.array(vec_loader.litmus_to_vec[name])
        else:
            # 如果缺失向量，保持为0或处理
            pass

    # 5. 切分矩阵
    train_idx = [litmus_dict[n] for n in train_names]
    test_idx = [litmus_dict[n] for n in test_names]

    for test_name in test_names:
        print(f"Testing {test_name}")
    X_train = full_feat[train_idx]
    y_train = full_perf[train_idx]

    X_test = full_feat[test_idx]
    y_test = full_perf[test_idx]

    # 6. (可选) 评估原始手工向量的 Baseline
    print("\n------------------------------------------------")
    print("【Baseline】原始手工特征 (13维) 直接计算距离：")
    analyze_embedding_quality(X_test, y_test, method_name="Raw Manual Features (Test Set)")

    # 7. 训练 DNN 映射
    print("\n>>> Training DNN to refine Manual Features...")
    emb_train, emb_test = run_train_and_inference(
        X_train, y_train, X_test,
        embedding_dim=16,
        epochs=500
    )

    # 8. 最终评估
    print("\n" + "=" * 60)
    print("【手工特征 + DNN 模型评估报告】")
    print("-" * 60)

    analyze_embedding_quality(emb_train, y_train, method_name="Train Set (Seen)")
    analyze_embedding_quality(emb_test, y_test, method_name="Test Set (Unseen)")

    print("=" * 60)

    # 9. 保存
    test_res = {name: emb_test[i].tolist() for i, name in enumerate(test_names)}
    with open("test_set_embeddings_manual.json", "w") as f:
        json.dump(test_res, f, indent=4)