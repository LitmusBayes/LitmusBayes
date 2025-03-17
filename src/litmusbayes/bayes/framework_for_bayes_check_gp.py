import json
import logging
import os
import random
import time
from collections import defaultdict

# === 引入 GPyTorch 和 PyTorch ===
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

# 引入评估指标和工具
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

import numpy as np

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


# ================= GPyTorch 模型定义 =================
# 使用稀疏变分高斯过程 (SVGP) 来处理大数据量
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list,
                 litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"):
        self.ps = param_space
        self.logger = get_logger(LOG_NAME)

        # 自动检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using Device: {self.device}")

        self.X = []
        self.y = []
        self.litmus_list = litmus_list

        # GP 对尺度非常敏感，必须使用 StandardScaler
        self.scaler = StandardScaler()

        self.model = None
        self.likelihood = None

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
        num_data = len(self.X)
        self.logger.info(f"Start fitting GPyTorch SVGP (N={num_data})...")

        # 1. 数据预处理
        X_arr = np.array(self.X)
        y_arr = np.array(self.y)

        # 特征标准化
        self.X_scaled = self.scaler.fit_transform(X_arr)

        # 目标值 Log 变换 (log1p)
        y_train_log = np.log1p(y_arr)

        # 转换为 Tensor 并移动到 GPU
        train_x = torch.tensor(self.X_scaled, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(y_train_log, dtype=torch.float32).to(self.device)

        # 2. 初始化诱导点 (Inducing Points)
        # 从训练数据中随机采样 500 个点作为诱导点，这大大降低了计算复杂度
        num_inducing = 500
        if num_data > num_inducing:
            # 随机选择索引
            inducing_idx = torch.randperm(num_data)[:num_inducing]
            inducing_points = train_x[inducing_idx].clone()
        else:
            inducing_points = train_x.clone()

        # 3. 初始化模型和似然
        self.model = SparseGPModel(inducing_points=inducing_points).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        # 4. 训练循环
        self.model.train()
        self.likelihood.train()

        # 优化器
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)

        # 损失函数 (Variational ELBO)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        # DataLoader 用于批量训练
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        epochs = 60  # 针对50000数据，50-60轮通常足够收敛
        start_time = time.time()

        for i in range(epochs):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 每 10 轮打印一次 Log
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                self.logger.info(f"Epoch {i + 1}/{epochs} - Loss: {avg_loss:.4f}")

        elapsed = time.time() - start_time
        self.logger.info(f"GPyTorch Fitting done. Time: {elapsed:.2f}s")

    def predict_one(self, litmus_name, param_vec):
        if self.model is None: return 0.0
        if litmus_name not in self.litmus_to_vector_dict:
            return None

        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)

        # 1. 预处理
        # 必须先 scaler transform，再转 tensor
        feature_scaled = self.scaler.transform(np.array([feature]))
        test_x = torch.tensor(feature_scaled, dtype=torch.float32).to(self.device)

        # 2. 预测模式
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.model(test_x)
            # 获取均值 (Log 域)
            pred_log = preds.mean.cpu().numpy()[0]

        # 3. 还原回去：exp(x) - 1
        return np.expm1(pred_log)


# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"

if __name__ == "__main__":
    # 1. Setup Logger
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 日志文件命名
    log_file_name = f"{stat_log_base}.gpytorch_run.log"
    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Evaluation Run (GPyTorch SVGP) | Seed={SEED} ===")

    # 2. 读取 Litmus List
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO (使用 GPyTorchBO)
    param_space = LitmusParamSpace()
    bo = GPyTorchBO(
        param_space,
        litmus_names,
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

    # 5. 切分数据
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 构建训练集 & 训练
    logger.info("Building training set...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])

    # 这里会调用 GPyTorch 的 Fit 流程
    bo.fit()

    # =========================================================
    # 7. 评估逻辑：Per-Litmus Top-1 Accuracy
    # =========================================================
    logger.info("Evaluating on test set (Per-Litmus Ranking Check)...")

    groups = defaultdict(list)
    y_true_all = []
    y_pred_all = []

    # 计时评估过程
    eval_start = time.time()

    for idx, item in enumerate(test_data):
        litmus = item["litmus"]
        param = item["param"]
        score = item["score"]

        pred = bo.predict_one(litmus, param)

        if pred is not None:
            record = {
                'param': param,
                'actual': score,
                'pred': pred
            }
            groups[litmus].append(record)

            y_true_all.append(score)
            y_pred_all.append(pred)

            # 打印进度
            if (idx + 1) % 1000 == 0:
                logger.info(f"Predicted {idx + 1}/{len(test_data)}...")
        else:
            logger.warning(f"[SKIP] #{idx} {litmus} (Missing vector)")

    eval_elapsed = time.time() - eval_start
    logger.info(f"Evaluation prediction finished in {eval_elapsed:.2f}s")

    # 统计逻辑保持不变
    total_litmus_cnt = 0
    top1_match_cnt = 0
    top3_match_cnt = 0

    logger.info("=" * 60)
    logger.info(
        f"{'LITMUS NAME':<30} | {'SAMPLES':<5} | {'TOP-1 MATCH?':<12} | {'ACTUAL BEST':<10} | {'MODEL PICK':<10}")
    logger.info("-" * 60)

    for litmus, records in groups.items():
        if len(records) < 2:
            continue

        total_litmus_cnt += 1

        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        best_actual_record = records_sorted_by_actual[0]
        max_actual_score = best_actual_record['actual']

        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        best_pred_record = records_sorted_by_pred[0]

        if max_actual_score == 1:
            total_litmus_cnt -= 1
            continue

        is_top1_correct = (best_pred_record['actual'] >= max_actual_score)

        if is_top1_correct:
            top1_match_cnt += 1
            match_str = "YES"
        else:
            match_str = "NO"

        top3_preds = records_sorted_by_pred[:3]
        is_top3_correct = any(r['actual'] >= max_actual_score for r in top3_preds)

        if is_top3_correct:
            top3_match_cnt += 1

        logger.info(
            f"{litmus[:30]:<30} | {len(records):<5} | {match_str:<12} | {max_actual_score:<10.2f} | {best_pred_record['actual']:<10.2f}")

    if total_litmus_cnt > 0:
        top1_acc = top1_match_cnt / total_litmus_cnt
        top3_acc = top3_match_cnt / total_litmus_cnt

        logger.info("=" * 60)
        logger.info("       PER-LITMUS RANKING RESULTS (GPyTorch SVGP)       ")
        logger.info("=" * 60)
        logger.info(f"Total Unique Litmus Tests: {total_litmus_cnt}")
        logger.info(f"Top-1 Accuracy:          {top1_acc * 100:.2f}% ({top1_match_cnt}/{total_litmus_cnt})")
        logger.info(f"Top-3 Recall:            {top3_acc * 100:.2f}% ({top3_match_cnt}/{total_litmus_cnt})")
        logger.info("-" * 60)

        y_true_all = np.array(y_true_all).reshape(-1)
        y_pred_all = np.array(y_pred_all).reshape(-1)
        res = spearmanr(y_true_all, y_pred_all)
        rho = res.statistic if hasattr(res, 'statistic') else res[0]
        logger.info(f"Global Spearman Rho:     {rho:.4f}")

        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        logger.info(f"Global R^2 Score:        {r2:.4f}")
        logger.info(f"Global MAE:              {mae:.4f}")
        logger.info("=" * 60)

    else:
        logger.warning("No litmus test groups with >1 samples found in test set.")

    per_litmus_rhos = []
    logger.info("-" * 60)
    logger.info("Calculating Per-Litmus Spearman Correlation...")

    for litmus, records in groups.items():
        if len(records) < 3:
            continue
        y_true_local = [r['actual'] for r in records]
        y_pred_local = [r['pred'] for r in records]

        if len(set(y_true_local)) <= 1 or len(set(y_pred_local)) <= 1:
            continue

        rho_local, _ = spearmanr(y_true_local, y_pred_local)
        if not np.isnan(rho_local):
            per_litmus_rhos.append(rho_local)

    if per_litmus_rhos:
        mean_rho = np.mean(per_litmus_rhos)
        median_rho = np.median(per_litmus_rhos)
        logger.info(f"Mean Per-Litmus Rho:   {mean_rho:.4f}")
        logger.info(f"Median Per-Litmus Rho: {median_rho:.4f}")
    else:
        logger.warning("Not enough data to calculate per-litmus Rho.")

    logger.info("=" * 60)