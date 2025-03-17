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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

# 复用你的工具类
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "dnn_eval"
BATCH_SIZE = 128
EPOCHS = 100  # 训练轮数
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. 定义 Dataset =================
class LitmusDataset(Dataset):
    def __init__(self, X, y):
        # 转换为 FloatTensor
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (N,) -> (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================= 2. 定义 DNN 模型 =================
class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        # 结构：Input -> [Linear-BN-ReLU-Dropout] -> [Linear-ReLU] -> Output
        # 针对 1.5w 数据量，不需要特别深，3层足够

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # 归一化，防止梯度消失/爆炸
            nn.ReLU(),
            nn.Dropout(0.2)  # 丢弃20%，防止过拟合
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.output = nn.Linear(32, 1)  # 回归输出 1 个值

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


# ================= 3. 封装训练流程 =================
class DNNRunner:
    def __init__(self, input_dim):
        self.model = SimpleDNN(input_dim).to(DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.logger = get_logger(LOG_NAME)
        self.scaler = StandardScaler()  # 用于特征归一化

    def fit(self, X_train, y_train):
        self.logger.info(f"Training on Device: {DEVICE}")

        # --- 关键步骤：特征归一化 ---
        # DNN 必须把输入缩放到 0 附近，否则很难收敛
        X_train_scaled = self.scaler.fit_transform(X_train)

        # --- 关键步骤：Target Log 变换 ---
        # 解决 Score 跨度过大问题 (e.g. 100 vs 50000)
        y_train_log = np.log1p(y_train)

        dataset = LitmusDataset(X_train_scaled, y_train_log)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    def predict(self, X_test):
        self.model.eval()
        # 记得用训练集的 scaler 进行 transform
        X_test_scaled = self.scaler.transform(X_test)
        X_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)

        with torch.no_grad():
            preds_log = self.model(X_tensor).cpu().numpy().flatten()

        # --- 还原：Log -> Exp ---
        return np.expm1(preds_log)


# ================= 4. 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"
cache_file_path = stat_log_base + ".cache_sum.jsonl"


def load_litmus_vectors(path):
    litmus_to_vec = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or ":" not in line: continue
            name, vec_str = line.split(":", 1)
            litmus_to_vec[name] = eval(vec_str)
    return litmus_to_vec


if __name__ == "__main__":
    # Setup
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger(f"{stat_log_base}.dnn.log", logging.INFO, LOG_NAME, stdout=True)
    logger.info("=== Start DNN Evaluation ===")

    # 1. 加载 Litmus 向量
    logger.info("Loading litmus vectors...")
    vec_dict = load_litmus_vectors(litmus_vec_path)

    # 2. 加载数据
    logger.info(f"Loading data from {cache_file_path} ...")
    raw_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        raw_data.append(json.loads(line))
                    except:
                        pass

    # 3. 构建特征矩阵 X 和 标签 y
    logger.info("Building feature matrix...")
    X_all = []
    y_all = []
    valid_data_indices = []

    for idx, item in enumerate(raw_data):
        l_name = item['litmus']
        p_vec = item['param']
        score = item['score']

        if l_name in vec_dict:
            l_vec = vec_dict[l_name]
            # 拼接: Param (24 dim) + LitmusVec (N dim)
            feature = list(p_vec) + list(l_vec)

            X_all.append(feature)
            y_all.append(score)
            valid_data_indices.append(idx)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    logger.info(f"Total valid samples: {len(X_all)} (Input Dim: {X_all.shape[1]})")

    # 4. 数据切分 (前 15000 训练，后 2000 测试)
    SPLIT_IDX = 20000

    if len(X_all) <= SPLIT_IDX:
        logger.error(f"Not enough data! Only {len(X_all)} samples, needed > {SPLIT_IDX}")
        exit(1)

    X_train, X_test = X_all[:SPLIT_IDX], X_all[SPLIT_IDX:]
    y_train, y_test = y_all[:SPLIT_IDX], y_all[SPLIT_IDX:]

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 5. 训练 DNN
    input_dim = X_train.shape[1]
    dnn = DNNRunner(input_dim)
    dnn.fit(X_train, y_train)

    # ================= 6. 预测 & 评估 (修复版) =================
    logger.info("Evaluating...")
    y_pred = dnn.predict(X_test)

    # --- 核心修复：强制转换为一维数组 ---
    # reshape(-1) 会把 (N, 1) 或 (1, N) 或 (N,) 统统变成 (N,)
    y_test_flat = np.array(y_test).reshape(-1)
    y_pred_flat = np.array(y_pred).reshape(-1)

    # 1. 计算 MSE, RMSE, MAE, R2
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    r2 = r2_score(y_test_flat, y_pred_flat)

    # 2. 安全计算 Spearman Rho
    res = spearmanr(y_test_flat, y_pred_flat)

    # 兼容处理：SciPy 不同版本返回值结构不同
    # 新版返回 object (.statistic), 旧版返回 tuple (correlation, pvalue)
    if hasattr(res, 'statistic'):
        rho = res.statistic
    else:
        rho = res[0]

    # 如果 rho 依然是被包裹在数组里 (比如 array(0.5))，提取出标量
    if hasattr(rho, 'item'):
        rho = rho.item()

    # 3. 记录日志
    logger.info("=" * 40)
    logger.info("       DNN RESULTS       ")
    logger.info("=" * 40)
    logger.info(f"Data Shapes: y_test={y_test_flat.shape}, y_pred={y_pred_flat.shape}")
    logger.info("-" * 40)
    logger.info(f"MSE  : {mse:.2f}")
    logger.info(f"RMSE : {rmse:.2f}")
    logger.info(f"MAE  : {mae:.2f}")
    logger.info(f"R2   : {r2:.4f}")
    logger.info(f"Rank Correlation (Rho): {rho:.4f}")
    logger.info("-" * 40)

