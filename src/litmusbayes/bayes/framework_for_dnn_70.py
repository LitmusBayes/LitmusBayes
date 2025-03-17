import json
import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# 复用你的工具类
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger

SEED = 2025
LOG_NAME = "dnn_eval"
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. 定义 Dataset =================
class LitmusDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================= 2. 定义 DNN 模型 (微调) =================
class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        # 使用 LeakyReLU 改善梯度流动
        # 增加层宽以容纳特征交互
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 1)  # 输出预测分数
        )

    def forward(self, x):
        return self.net(x)


# ================= 3. 封装训练流程 =================
class DNNRunner:
    def __init__(self, input_dim):
        self.model = SimpleDNN(input_dim).to(DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.logger = get_logger(LOG_NAME)
        self.scaler_X = StandardScaler()
        # 我们可以选择是否对 Y 也就是 Score 进行归一化，这里暂时只做 Log 处理

    def fit(self, X_train, y_train):
        self.logger.info(f"Training on Device: {DEVICE}")

        # 归一化特征
        X_train_scaled = self.scaler_X.fit_transform(X_train)

        # Log 变换 Target (解决数值跨度大问题)
        y_train_log = np.log1p(y_train)

        dataset = LitmusDataset(X_train_scaled, y_train_log)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    def predict(self, X_test):
        self.model.eval()
        X_test_scaled = self.scaler_X.transform(X_test)
        X_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)

        with torch.no_grad():
            preds_log = self.model(X_tensor).cpu().numpy().flatten()

        # 还原 Log
        return np.expm1(preds_log)


# ================= 4. 数据处理工具 (核心修改) =================

def load_data_grouped_by_test(cache_path, vec_path):
    """
    加载数据，并按 Litmus Test Name 进行分组。
    Returns:
        dict: { 'test_name': [ {'feature': [...], 'score': float}, ... ] }
    """
    # 1. 加载向量字典
    litmus_to_vec = {}
    with open(vec_path, "r") as f:
        for line in f:
            if not line.strip() or ":" not in line: continue
            name, vec_str = line.split(":", 1)
            litmus_to_vec[name] = eval(vec_str)

    # 2. 加载原始数据并分组
    grouped_data = {}
    logger = get_logger(LOG_NAME)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    l_name = item['litmus']
                    p_vec = item['param']
                    score = item['score']

                    if l_name in litmus_to_vec:
                        l_vec = litmus_to_vec[l_name]
                        # 组合特征: Param + Code Vector
                        feature = list(p_vec) + list(l_vec)

                        if l_name not in grouped_data:
                            grouped_data[l_name] = []
                        grouped_data[l_name].append({'feature': feature, 'score': score})
                except Exception as e:
                    continue

    return grouped_data


def flatten_data(grouped_data, test_names):
    """将选定的 test_names 的数据展平为 X, y 用于训练"""
    X, y = [], []
    for name in test_names:
        for item in grouped_data[name]:
            X.append(item['feature'])
            y.append(item['score'])
    return np.array(X), np.array(y)


# ================= 5. Ranking 评估逻辑 =================

def evaluate_ranking_capability(model, grouped_data, test_names_set):
    """
    按 Litmus Test 逐个评估：模型是否能把最好的参数排在前面？
    """
    logger = get_logger(LOG_NAME)
    logger.info("Starting Ranking Evaluation (Top-K Analysis)...")

    top1_hits = 0  # 预测的第一名确实是真实的第一名
    top3_hits = 0  # 预测的第一名在真实的前三名里
    top5_hits = 0  # 预测的第一名在真实的前五名里
    total_tests = 0

    rhos = []  # 斯皮尔曼相关系数列表

    for name in test_names_set:
        samples = grouped_data[name]
        if len(samples) < 5: continue  # 样本太少不测

        total_tests += 1

        # 提取该 Test 的所有数据
        X_test_local = np.array([s['feature'] for s in samples])
        y_true_local = np.array([s['score'] for s in samples])

        # 预测
        y_pred_local = model.predict(X_test_local)

        # --- 核心：排序分析 ---
        # 获取真实分数的降序索引 (假设分数越高越好)
        true_ranks = np.argsort(y_true_local)[::-1]
        # 获取预测分数的降序索引
        pred_ranks = np.argsort(y_pred_local)[::-1]

        # 1. 我们的模型推荐了哪个参数？(预测分最高的那个的索引)
        model_recommendation_idx = pred_ranks[0]

        # 2. 这个推荐参数，在真实情况里排第几？
        # np.where 返回的是 tuple，取 [0][0]
        real_rank_of_recommendation = np.where(true_ranks == model_recommendation_idx)[0][0]

        if real_rank_of_recommendation == 0:
            top1_hits += 1
        if real_rank_of_recommendation < 3:
            top3_hits += 1
        if real_rank_of_recommendation < 5:
            top5_hits += 1

        # 3. 计算相关性
        rho, _ = spearmanr(y_true_local, y_pred_local)
        if not np.isnan(rho):
            rhos.append(rho)

    logger.info("-" * 40)
    logger.info(f"Evaluated on {total_tests} unseen Litmus Tests")
    logger.info(f"Avg Spearman Correlation : {np.mean(rhos):.4f}")
    logger.info("-" * 40)
    logger.info(f"Top-1 Accuracy: {top1_hits / total_tests * 100:.2f}% (Model picked the absolute best)")
    logger.info(f"Top-3 Accuracy: {top3_hits / total_tests * 100:.2f}% (Model picked one of top 3)")
    logger.info(f"Top-5 Accuracy: {top5_hits / total_tests * 100:.2f}% (Model picked one of top 5)")
    logger.info("-" * 40)
    logger.info("Interpretation: Top-3 Accuracy is the most important metric for Auto-tuning.")


# ================= 6. 主程序 =================
# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"
cache_file_path = stat_log_base + ".cache_sum_70_no.jsonl"

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger(f"{stat_log_base}.dnn.log", logging.INFO, LOG_NAME, stdout=True)
    logger.info("=== Start DNN Evaluation (Test-wise Split) ===")

    # 1. 加载并分组数据
    grouped_data = load_data_grouped_by_test(cache_file_path, litmus_vec_path)
    all_test_names = list(grouped_data.keys())
    random.shuffle(all_test_names)  # 随机打乱 Test 名字列表

    logger.info(f"Total unique Litmus Tests: {len(all_test_names)}")

    # 2. 按 Litmus Test 进行切分 (关键修改)
    # 80% 的 Tests 用于训练，20% 的 Tests 用于完全不可见的验证
    split_point = int(len(all_test_names) * 0.8)
    train_names = all_test_names[:split_point]
    test_names = all_test_names[split_point:]

    logger.info(f"Train Tests: {len(train_names)} | Test Tests: {len(test_names)}")
    from sklearn.model_selection import train_test_split

    # 把所有数据先展平
    X_all, y_all = flatten_data(grouped_data, all_test_names)

    # 随机切分 80/20
    X_train, X_test_flat, y_train, y_test_flat = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, shuffle=True
    )
    # 3. 展平数据用于 DNN 训练
    # X_train, y_train = flatten_data(grouped_data, train_names)
    # X_test_flat, y_test_flat = flatten_data(grouped_data, test_names)  # 仅用于算MSE，不用于Ranking评估

    logger.info(f"Train Samples: {len(X_train)} | Test Samples: {len(X_test_flat)}")

    # 4. 训练
    input_dim = X_train.shape[1]
    dnn = DNNRunner(input_dim)
    dnn.fit(X_train, y_train)

    # 5. 传统指标评估 (MSE/R2)
    logger.info("Evaluating Global Metrics (MSE/R2)...")
    y_pred_flat = dnn.predict(X_test_flat)

    # 转换为 1D 数组
    y_test_flat = y_test_flat.reshape(-1)
    y_pred_flat = y_pred_flat.reshape(-1)

    r2 = 0
    try:
        from sklearn.metrics import r2_score

        r2 = r2_score(y_test_flat, y_pred_flat)
    except:
        pass

    logger.info(f"Global R2 Score: {r2:.4f}")

    # 6. 高级指标评估 (Ranking Capability) - 这才是你的论文需要的
    evaluate_ranking_capability(dnn, grouped_data, test_names)