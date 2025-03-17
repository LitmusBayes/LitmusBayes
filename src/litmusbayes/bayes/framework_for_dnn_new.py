import json
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# 引入你的工具库 (保持不变)
from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval_nn"


# ================= PyTorch 模型定义 =================

class SensitivityEmbeddingModel(nn.Module):
    def __init__(self, num_tests, param_dim=11, embedding_dim=32):
        super(SensitivityEmbeddingModel, self).__init__()

        # 1. 潜在特征 (Latent Embedding)
        # 我们依然保留 32 维，因为这能容纳更丰富的信息
        self.test_embedding = nn.Embedding(num_tests, embedding_dim)

        # 2. 敏感度生成器 (Sensitivity Generator)
        # 将 32 维的特征映射为 11 维的“参数权重”
        # 这就是让模型根据 Test ID 决定：这一组 Test 对哪几个参数最敏感？
        self.to_sensitivity = nn.Sequential(
            nn.Linear(embedding_dim, param_dim),
            nn.Tanh()  # 限制在 -1 到 1 之间，表示正相关/负相关/不相关
        )

        # 3. 偏差生成器 (Bias Generator)
        # 还是需要一个 bias 来处理基础分数的不同
        self.to_bias = nn.Linear(embedding_dim, param_dim)

        # 4. 推理网络 (Shared Reasoning Network)
        # 输入维度依然是 11，但经过了调制
        self.net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, test_ids, params):
        # 1. 获取该 Test 的潜在特征
        emb = self.test_embedding(test_ids)  # [Batch, 32]

        # 2. 生成敏感度掩码 (Gate) 和 偏差 (Bias)
        # gate: 表示该 Test 对每个参数的“重视程度”
        sensitivity_gate = self.to_sensitivity(emb)  # [Batch, 11]
        bias = self.to_bias(emb)  # [Batch, 11]

        # 3. 核心步骤：特征调制 (Feature Modulation)
        # 公式: y = x * w + b
        # 如果某个参数变化引起数值剧烈波动，模型必须把对应的 gate 权值学大
        modulated_params = params * sensitivity_gate + bias

        # 4. 输入后续网络
        return self.net(modulated_params)

# ================= 封装训练逻辑的类 =================

class NeuralNetPredictor:
    def __init__(self, litmus_list, embedding_dim=32, epochs=50, batch_size=64, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger(LOG_NAME)

        # 建立 Litmus Name 到 ID 的映射
        self.litmus_to_id = {name: i for i, name in enumerate(litmus_list)}
        self.id_to_litmus = {i: name for name, i in self.litmus_to_id.items()}

        self.model = SensitivityEmbeddingModel(
            num_tests=len(litmus_list),
            param_dim=11,
            embedding_dim=embedding_dim
        ).to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # 数据容器
        self.raw_params = []
        self.raw_ids = []
        self.raw_y = []

        # 归一化工具 (神经网络必须做 Feature Scaling)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_id:
            return

        # 存储原始数据
        self.raw_ids.append(self.litmus_to_id[litmus_name])
        self.raw_params.append(list(param_vec))
        self.raw_y.append(score)

    def fit(self):
        self.logger.info(f"Start Neural Network training on {self.device}...")

        # 1. 数据预处理
        X_params = np.array(self.raw_params)
        y = np.array(self.raw_y)

        # 对参数进行标准化 (Mean=0, Std=1)
        X_params_scaled = self.scaler.fit_transform(X_params)

        # 对 Target 进行 Log1p 处理 (和你之前的逻辑一致)
        y_log = np.log1p(y)

        # 转换为 Tensor
        tensor_ids = torch.LongTensor(self.raw_ids)
        tensor_params = torch.FloatTensor(X_params_scaled)
        tensor_y = torch.FloatTensor(y_log).view(-1, 1)

        dataset = TensorDataset(tensor_ids, tensor_params, tensor_y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2. 定义优化器和 Loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # 3. 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for b_ids, b_params, b_y in dataloader:
                b_ids, b_params, b_y = b_ids.to(self.device), b_params.to(self.device), b_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(b_ids, b_params)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs} | Loss: {avg_loss:.6f}")

        self.is_fitted = True

    def predict_one(self, litmus_name, param_vec):
        if not self.is_fitted:
            raise Exception("Model not fitted yet!")
        if litmus_name not in self.litmus_to_id:
            return None

        self.model.eval()
        with torch.no_grad():
            # 准备数据
            test_id = torch.LongTensor([self.litmus_to_id[litmus_name]]).to(self.device)

            # 参数归一化
            param_vec_scaled = self.scaler.transform([list(param_vec)])
            param_tensor = torch.FloatTensor(param_vec_scaled).to(self.device)

            # 预测
            pred_log = self.model(test_id, param_tensor).item()

            # 还原: expm1
            return max(0, np.expm1(pred_log))  # 确保非负

    def save_embeddings(self, output_path):
        """
        训练完成后，将学到的 Embedding 导出，这就是 Litmus Test 的相似性向量
        """
        self.logger.info(f"Saving learned embeddings to {output_path}...")
        embeddings = self.model.test_embedding.weight.data.cpu().numpy()

        with open(output_path, "w") as f:
            for litmus_name, idx in self.litmus_to_id.items():
                vec = embeddings[idx].tolist()
                # 保存格式: NAME: [v1, v2, ...]
                f.write(f"{litmus_name}:{vec}\n")


# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
cache_file_path = stat_log_base + ".cache.cleaned.jsonl"
# 输出的新向量路径
output_embedding_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/learned_embeddings.txt"

if __name__ == "__main__":
    # 1. Setup
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger(
        log_file=f"{stat_log_base}.nn_run.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Joint Embedding NN Run | Seed={SEED} ===")

    # 2. 读取 Litmus List
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 NN Predictor (替代原来的 Random Forest BO)
    # embedding_dim=32 表示我们将每个 Litmus Test 压缩成 32 维向量
    nn_predictor = NeuralNetPredictor(
        litmus_names,
        embedding_dim=32,
        epochs=50,  # 训练轮数，可调整
        lr=0.005  # 学习率
    )

    # 4. 加载 Cache 数据
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
    # 5. 切分数据
    train_data = all_data[:10000]
    test_data = all_data[10000:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 构建训练集 & 训练
    logger.info("Building training set for Neural Network...")
    for item in train_data:
        nn_predictor.add(item["litmus"], item["param"], item["score"])

    # 开始训练网络 (会自动学习 Embedding)
    nn_predictor.fit()

    # 【重要】导出学习到的 Embedding，这就是你做相似性分析需要的东西
    nn_predictor.save_embeddings(output_embedding_path)

    # =========================================================
    # 7. 评估逻辑 (逻辑保持不变，只是调用对象变成了 nn_predictor)
    # =========================================================
    logger.info("Evaluating on test set (Per-Litmus Ranking Check)...")

    groups = defaultdict(list)
    y_true_all = []
    y_pred_all = []

    for idx, item in enumerate(test_data):
        litmus = item["litmus"]
        param = item["param"]
        score = item["score"]

        # 使用 NN 预测
        pred = nn_predictor.predict_one(litmus, param)

        if pred is not None:
            record = {'param': param, 'actual': score, 'pred': pred}
            groups[litmus].append(record)
            y_true_all.append(score)
            y_pred_all.append(pred)

    total_litmus_cnt = 0
    top1_match_cnt = 0
    top3_match_cnt = 0

    logger.info("-" * 80)
    logger.info(f"{'LITMUS NAME':<30} | {'CNT':<3} | {'TOP1?':<5} | {'ACTUAL BEST':<10} | {'MODEL PICK':<10}")
    logger.info("-" * 80)

    for litmus, records in groups.items():
        if len(records) < 2: continue
        total_litmus_cnt += 1

        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        best_actual_record = records_sorted_by_actual[0]
        max_actual_score = best_actual_record['actual']

        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        best_pred_record = records_sorted_by_pred[0]

        if max_actual_score == 0:
            total_litmus_cnt -= 1
            continue

        is_top1_correct = (best_pred_record['actual'] >= max_actual_score)
        if is_top1_correct: top1_match_cnt += 1

        top3_preds = records_sorted_by_pred[:3]
        is_top3_correct = any(r['actual'] >= max_actual_score for r in top3_preds)
        if is_top3_correct: top3_match_cnt += 1

        match_str = "YES" if is_top1_correct else "NO"
        # 限制打印数量防止刷屏，或者根据需要保留
        if total_litmus_cnt <= 20 or not is_top1_correct:
            logger.info(
                f"{litmus[:30]:<30} | {len(records):<3} | {match_str:<5} | {max_actual_score:<10.2f} | {best_pred_record['actual']:<10.2f}")

    if total_litmus_cnt > 0:
        top1_acc = top1_match_cnt / total_litmus_cnt
        top3_acc = top3_match_cnt / total_litmus_cnt

        logger.info("=" * 60)
        logger.info("       PER-LITMUS RANKING RESULTS (NN)       ")
        logger.info("=" * 60)
        logger.info(f"Total Valid Tests:       {total_litmus_cnt}")
        logger.info(f"Top-1 Accuracy:          {top1_acc * 100:.2f}%")
        logger.info(f"Top-3 Recall:            {top3_acc * 100:.2f}%")

        y_true_all = np.array(y_true_all).reshape(-1)
        y_pred_all = np.array(y_pred_all).reshape(-1)
        res = spearmanr(y_true_all, y_pred_all)
        rho = res.statistic if hasattr(res, 'statistic') else res[0]
        logger.info(f"Global Spearman Rho:     {rho:.4f}")
        logger.info(f"Embeddings saved to:     {output_embedding_path}")
        logger.info("=" * 60)