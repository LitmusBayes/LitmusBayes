import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "nn_eval"
BATCH_SIZE = 256  # 稍微调大 Batch Size 加速评估
EPOCHS = 100  # 增加轮数确保收敛
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. 数据集定义 =================

class LitmusDataset(Dataset):
    def __init__(self, X_litmus, X_param, y):
        self.X_litmus = torch.FloatTensor(X_litmus)
        self.X_param = torch.FloatTensor(X_param)
        self.y = torch.FloatTensor(y).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_litmus[idx], self.X_param[idx], self.y[idx]


# ================= 2. 双塔模型定义 =================
#

class TwoTowerNet(nn.Module):
    def __init__(self, litmus_dim, param_dim, embedding_dim=64):
        super(TwoTowerNet, self).__init__()

        # --- 左塔: 代码特征 ---
        self.litmus_tower = nn.Sequential(
            nn.Linear(litmus_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

        # --- 右塔: 参数特征 ---
        self.param_tower = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )

        # --- 融合层: 显式交互 ---
        # 拼接(Concat) + 点乘(Dot Product)
        fusion_input_dim = embedding_dim * 2 + embedding_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_litmus, x_param):
        emb_litmus = self.litmus_tower(x_litmus)
        emb_param = self.param_tower(x_param)

        # 显式构造交互特征
        concat_feat = torch.cat([emb_litmus, emb_param], dim=1)
        interact_feat = emb_litmus * emb_param  # 核心：强制交互

        combined = torch.cat([concat_feat, interact_feat], dim=1)
        out = self.fusion_layer(combined)
        return out


# ================= 3. 神经网络 BO 封装 =================

class NeuralNetBO:
    def __init__(self, litmus_vec_path):
        self.logger = get_logger(LOG_NAME)
        self.litmus_to_vec = self.load_litmus_vectors(litmus_vec_path)

        self.model = None
        # 定义 Scaler，用于标准化输入
        self.l_scaler = StandardScaler()
        self.p_scaler = StandardScaler()

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line: continue
                    name, vec_str = line.split(":", 1)
                    try:
                        vec = eval(vec_str)
                        litmus_to_vec[name] = list(vec)
                    except:
                        pass
        return litmus_to_vec

    def prepare_xy(self, data_list, fit_scaler=False):
        """将 JSON 对象列表转换为 numpy 数组"""
        X_l, X_p, y = [], [], []
        valid_indices = []

        for i, item in enumerate(data_list):
            name = item['litmus']
            if name not in self.litmus_to_vec: continue

            X_l.append(self.litmus_to_vec[name])
            X_p.append(item['param'])
            y.append(item['score'])
            valid_indices.append(i)

        X_l = np.array(X_l)
        X_p = np.array(X_p)
        y = np.array(y)

        if fit_scaler and len(X_l) > 0:
            X_l = self.l_scaler.fit_transform(X_l)
            X_p = self.p_scaler.fit_transform(X_p)
        elif len(X_l) > 0:
            X_l = self.l_scaler.transform(X_l)
            X_p = self.p_scaler.transform(X_p)

        return X_l, X_p, y, valid_indices

    def fit(self, train_data):
        self.logger.info("Preprocessing training data...")
        X_l, X_p, y, _ = self.prepare_xy(train_data, fit_scaler=True)

        # Log1p 变换
        y_log = np.log1p(y)

        # DataLoader
        dataset = LitmusDataset(X_l, X_p, y_log)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Init Model
        litmus_dim = X_l.shape[1]
        param_dim = X_p.shape[1]
        self.model = TwoTowerNet(litmus_dim, param_dim).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        self.logger.info(f"Start Training on {DEVICE} (Epochs={EPOCHS})...")

        self.model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for xl, xp, target in loader:
                xl, xp, target = xl.to(DEVICE), xp.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = self.model(xl, xp)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss / len(loader):.4f}")

    def predict_batch(self, litmus_list, param_list):
        """
        批量预测接口，与 RF 保持一致
        """
        # 构造临时 data list 以复用 prepare_xy
        temp_data = [{'litmus': l, 'param': p, 'score': 0} for l, p in zip(litmus_list, param_list)]

        X_l, X_p, _, valid_indices = self.prepare_xy(temp_data, fit_scaler=False)

        if len(X_l) == 0:
            return [], []

        # 转 Tensor
        dataset = LitmusDataset(X_l, X_p, np.zeros(len(X_l)))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

        self.model.eval()
        all_preds_log = []

        with torch.no_grad():
            for xl, xp, _ in loader:
                xl, xp = xl.to(DEVICE), xp.to(DEVICE)
                out = self.model(xl, xp)
                all_preds_log.extend(out.cpu().numpy().flatten())

        # 还原 Expm1
        return np.expm1(all_preds_log), valid_indices


# ================= 4. 评估工具函数 (Top-K, Ranking) =================

def analyze_ranking_quality(groups_data):
    regret_list = []

    for litmus, records in groups_data.items():
        if len(records) < 2: continue

        # Ground Truth Best
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        gt_best_val = records_sorted_by_actual[0]['actual']
        if gt_best_val <= 0: continue

        # Model Prediction Best
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        model_pick_val = records_sorted_by_pred[0]['actual']

        regret = (gt_best_val - model_pick_val) / gt_best_val
        regret_list.append(regret)

    print("\n" + "=" * 40)
    print("      NEURAL NET DIAGNOSIS REPORT      ")
    print("=" * 40)

    regret_arr = np.array(regret_list)
    print(f"Total Groups: {len(regret_list)}")
    print(f"Mean Regret:  {np.mean(regret_arr) * 100:.2f}%")
    print(f"Zero Regret:  {np.mean(regret_arr == 0) * 100:.2f}% (Perfect Match)")


def analyze_topk_recall(groups_data):
    k_values = [1, 3, 5, 10]
    hits = {k: 0 for k in k_values}
    total_groups = 0

    for litmus, records in groups_data.items():
        if len(records) < 2: continue
        total_groups += 1

        max_actual = max(r['actual'] for r in records)
        sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)

        for k in k_values:
            candidates = sorted_by_pred[:k]
            if any(c['actual'] >= max_actual for c in candidates):
                hits[k] += 1

    print("\n" + "=" * 40)
    print("      TOP-K RECALL ANALYSIS      ")
    print("=" * 40)
    for k in k_values:
        print(f"Top-{k:<2} Recall: {hits[k] / total_groups * 100:.2f}%")


# ================= 主程序 =================

if __name__ == "__main__":
    # 1. Setup
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
    litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
    cache_file_path = stat_log_base + ".cache4_norm.jsonl"

    log_file_name = f"{stat_log_base}.nn_eval.run.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Neural Net Evaluation | Seed={SEED} ===")

    # 2. 加载数据
    logger.info("Loading Data...")
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass

    # 3. 切分数据 (Train / Test)
    # 关键点：必须确保切分逻辑和 RF 一致，这样结果才可比
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.7)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    logger.info(f"Train Size: {len(train_data)} | Test Size: {len(test_data)}")

    # 4. 训练神经网络
    nn_bo = NeuralNetBO(litmus_vec_path)
    nn_bo.fit(train_data)

    # 5. 在测试集上批量预测
    logger.info("Predicting on Test Set...")
    test_litmus_names = [item['litmus'] for item in test_data]
    test_params = [item['param'] for item in test_data]

    t0 = time.time()
    preds, valid_indices = nn_bo.predict_batch(test_litmus_names, test_params)
    logger.info(f"Prediction Done in {time.time() - t0:.2f}s")

    # 6. 整理结果并评估
    groups = defaultdict(list)
    y_true_all = []
    y_pred_all = []

    for ptr, original_idx in enumerate(valid_indices):
        item = test_data[original_idx]
        actual_score = item['score']
        pred_score = preds[ptr]

        record = {'param': item['param'], 'actual': actual_score, 'pred': pred_score}
        groups[item['litmus']].append(record)

        y_true_all.append(actual_score)
        y_pred_all.append(pred_score)

    # 7. 打印评估报告
    if len(y_true_all) > 0:
        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        logger.info(f"Global R^2 Score: {r2:.4f}")
        logger.info(f"Global MAE:       {mae:.4f}")

        analyze_ranking_quality(groups)
        analyze_topk_recall(groups)
    else:
        logger.error("No valid predictions generated!")