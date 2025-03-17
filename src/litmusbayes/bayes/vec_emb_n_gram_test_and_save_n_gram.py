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

from src.slide.bayes.litmus_param_space import LitmusParamSpace
from src.slide.bayes.logger_util import setup_logger, get_logger
from src.slide.bayes.util import get_files

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
from sklearn.feature_extraction.text import CountVectorizer
import re


# ==========================================
# 1. 特征提取 (修改：增加了 transform 方法)
# ==========================================
class LitmusFeatureExtractor:
    def __init__(self, ngram_range=(1, 3)):
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            token_pattern=r'(?u)\b\w[\w.]*\b',
            lowercase=False,
            max_features=2000
        )
        self.feature_names = None

    def parse_raw_lines(self, raw_lines):
        parsed_data = {}
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].strip()
                body = parts[1].strip()
                clean_body = body.replace("<SEP>", " SEP ")
                clean_body = re.sub(r'\s+', ' ', clean_body).strip()
                parsed_data[name] = clean_body
        return parsed_data

    def _prepare_corpus(self, litmus_name_list, raw_structure_dict):
        """内部辅助函数：准备语料"""
        corpus_segmented = []
        for name in litmus_name_list:
            if name in raw_structure_dict:
                threads = raw_structure_dict[name].split("SEP")
                corpus_segmented.extend(threads)
            else:
                corpus_segmented.append("")
        return corpus_segmented

    def _aggregate_vectors(self, litmus_name_list, raw_structure_dict, transformed_vectors):
        """内部辅助函数：聚合线程向量"""
        final_X = []
        if hasattr(transformed_vectors, "toarray"):
            transformed_vectors = transformed_vectors.toarray()

        cursor = 0
        for name in litmus_name_list:
            if name in raw_structure_dict:
                threads = raw_structure_dict[name].split("SEP")
                num_threads = len(threads)
                thread_vectors = transformed_vectors[cursor: cursor + num_threads]
                # (Vec1 + 1) * (Vec2 + 1) ...
                sample_vector = np.prod(thread_vectors + 1, axis=0)
                final_X.append(sample_vector)
                cursor += num_threads
            else:
                final_X.append(np.zeros(len(self.feature_names)))
                cursor += 1
        return np.array(final_X)

    def fit_transform(self, litmus_name_list, raw_structure_dict):
        # 1. 准备语料
        corpus = self._prepare_corpus(litmus_name_list, raw_structure_dict)
        # 2. 训练并转换
        vecs = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        # 3. 聚合
        return self._aggregate_vectors(litmus_name_list, raw_structure_dict, vecs)

    def transform(self, litmus_name_list, raw_structure_dict):
        """
        [新增] 仅转换，不训练。用于推理新数据。
        """
        if self.feature_names is None:
            raise ValueError("Vectorizer has not been fitted yet!")

        # 1. 准备语料
        corpus = self._prepare_corpus(litmus_name_list, raw_structure_dict)
        # 2. 仅转换 (使用已有的词表)
        vecs = self.vectorizer.transform(corpus)
        # 3. 聚合
        return self._aggregate_vectors(litmus_name_list, raw_structure_dict, vecs)

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
    X_train: (N_train, FeatDim)
    y_train: (N_train, ParamDim)
    X_test:  (N_test, FeatDim)
    """
    # === 1. 数据归一化 (仅使用训练集统计量) ===
    # 避免测试集信息泄露
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # === 2. 准备 Tensor ===
    xt_tensor = torch.FloatTensor(X_train_norm)
    yt_tensor = torch.FloatTensor(np.log1p(y_train))  # Log1p 变换 Target
    xv_tensor = torch.FloatTensor(X_test_norm)

    # DataLoader
    dataset = TensorDataset(xt_tensor, yt_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # === 3. 初始化模型 ===
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LitmusEmbedder(input_dim, output_dim, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = PearsonMSELoss(alpha=0.7).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    print(f"\n>>> Start Training (Device: {device})...")

    # === 4. 训练循环 ===
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

    # === 5. 推理 (生成 Embedding) ===
    print(">>> Generating embeddings for Train and Test sets...")
    model.eval()

    with torch.no_grad():
        _, emb_train = model(xt_tensor.to(device))
        emb_train = emb_train.cpu().numpy()

        _, emb_test = model(xv_tensor.to(device))
        emb_test = emb_test.cpu().numpy()

    # === 6. [重要] 返回模型和归一化参数，以便后续推理新数据 ===
    return emb_train, emb_test, model, X_mean, X_std


# ==========================================
# 6. 新增：新数据推理与保存函数
# ==========================================
def generate_embeddings_for_new_files(
        trained_model,
        feature_extractor,
        train_mean,
        train_std,
        new_text_path,
        output_file_path
):
    """
    使用训练好的模型和特征提取器，处理全新的文件并保存向量
    """
    print(f"\n>>> 开始处理新数据推理 (N-gram + DNN)...")
    print(f"Text File: {new_text_path}")

    if not os.path.exists(new_text_path):
        print(f"Error: Text file not found {new_text_path}")
        return

    # 1. 读取新文件
    with open(new_text_path, "r") as f:
        raw_lines = f.readlines()

    # 2. 解析结构
    # parse_raw_lines 返回的是 dict {name: body}
    new_structure_map = feature_extractor.parse_raw_lines(raw_lines)
    new_names_list = sorted(list(new_structure_map.keys()))

    if len(new_names_list) == 0:
        print("No valid litmus tests found in new file.")
        return

    print(f"Found {len(new_names_list)} unique tests in new file.")

    # 3. 提取特征 (Transform Only)
    print("Extracting N-gram features (using trained vocabulary)...")
    X_new_raw = feature_extractor.transform(new_names_list, new_structure_map)

    # 4. 归一化 (关键！使用训练集的参数)
    print("Normalizing data using training statistics...")
    X_new_norm = (X_new_raw - train_mean) / train_std

    # 5. 模型推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()

    # 转 Tensor
    x_tensor = torch.FloatTensor(X_new_norm).to(device)

    # DataLoader 处理 (防止数据量大爆显存)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_embeddings = []
    print("Running DNN inference...")
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0]  # TensorDataset 返回 tuple
            _, emb = trained_model(b_x)
            all_embeddings.append(emb.cpu().numpy())

    final_embeddings = np.vstack(all_embeddings)

    # 6. 保存结果
    print(f"Saving results to {output_file_path} ...")
    with open(output_file_path, "w") as f_out:
        for i, name in enumerate(new_names_list):
            vec = X_new_raw[i].tolist()
            # 格式化: Name:[1.0, 2.0, ...]
            vec_str = json.dumps(vec)
            f_out.write(f"{name}:{vec_str}\n")

    print("Done.")

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"
litmus_cycle_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3.log"


inference_text_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3_aligned.log" # 请修改为你的新文件路径
inference_output_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_ngram_naive.log"

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger(f"{stat_log_base}.check.run.log", level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start N-gram DNN Split Evaluation | Seed={SEED} ===")

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
    logger.info(f"Total Unique Litmus Tests: {len(all_litmus_list)}")

    # 2. 【核心修改】划分训练集和测试集 (按名字划分)
    train_names, test_names = train_test_split(all_litmus_list, test_size=0.20, random_state=SEED)
    for test_name in test_names:
        print(test_name)
    logger.info(f"Train Set Size: {len(train_names)}")
    logger.info(f"Test Set Size:  {len(test_names)}")

    # 3. 建立全局索引和矩阵
    # 参数列统一
    all_params = sorted(list(set([str(item["param"]) for item in all_raw_data])))
    param_dict = {p: i for i, p in enumerate(all_params)}
    litmus_dict = {name: i for i, name in enumerate(all_litmus_list)}

    # 构建 Full Performance Matrix
    full_performance = np.zeros((len(all_litmus_list), len(all_params)))
    for item in all_raw_data:
        l_idx = litmus_dict[item["litmus"]]
        p_idx = param_dict[str(item["param"])]
        full_performance[l_idx][p_idx] = item["score"]

    # 4. 提取 N-gram 特征 (对全量 Litmus 提取以保证维度对齐)
    logger.info("Extracting N-gram features...")
    if os.path.exists(litmus_cycle_path):
        with open(litmus_cycle_path, "r") as f:
            raw_lines = f.readlines()
        feature_extractor = LitmusFeatureExtractor(ngram_range=(1, 3))
        structure_map = feature_extractor.parse_raw_lines(raw_lines)

        # 传入所有 Litmus 名字，确保顺序与 full_performance 一致
        full_X_ngram = feature_extractor.fit_transform(all_litmus_list, structure_map)
    else:
        logger.error("Litmus text file missing!")
        exit(1)

    # 5. 根据名字划分数据矩阵
    train_indices = [litmus_dict[n] for n in train_names]
    test_indices = [litmus_dict[n] for n in test_names]

    X_train = full_X_ngram[train_indices]
    y_train = full_performance[train_indices]

    X_test = full_X_ngram[test_indices]
    y_test = full_performance[test_indices]
    analyze_embedding_quality(X_train, y_train, method_name="Raw N-gram Features(Train)")
    analyze_embedding_quality(X_test, y_test, method_name="Raw N-gram Features(Test)")
    # 6. 运行训练和推理
    logger.info("Training DNN with N-gram features...")
    emb_train, emb_test, trained_model, train_mean, train_std = run_train_and_inference(
        X_train, y_train, X_test,
        embedding_dim=16,
        epochs=500
    )

    # 7. 评估结果
    print("\n" + "=" * 60)
    print("【N-gram + DNN 模型评估报告】")
    print("-" * 60)

    # 评估训练集
    analyze_embedding_quality(emb_train, y_train, method_name="Train Set (Seen)")

    # 评估测试集 (关键指标)
    analyze_embedding_quality(emb_test, y_test, method_name="Test Set (Unseen)")

    print("=" * 60)

    # 8. 保存测试集结果

    NEW_TEXT_PATH = inference_text_path
    OUTPUT_FILE = inference_output_file

    generate_embeddings_for_new_files(
        trained_model,
        feature_extractor,
        train_mean,
        train_std,
        NEW_TEXT_PATH,
        OUTPUT_FILE
    )