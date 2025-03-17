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
import re
from collections import defaultdict

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer

# PyG
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split
# Custom Imports (保留你的引用)
from src.litmusbayes.bayes.logger_util import setup_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ==========================================
# 1. N-gram 特征提取器 (来自你的代码)
# ==========================================
class LitmusFeatureExtractor:
    def __init__(self, ngram_range=(1, 3)):
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            token_pattern=r'(?u)\b\w[\w.]+\b',
            lowercase=False,
            max_features=2000  # 限制一下维度，防止 OOM，通常前2000个最关键
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

    def fit_transform(self, litmus_name_list, raw_structure_dict):
        corpus = []
        for name in litmus_name_list:
            if name in raw_structure_dict:
                corpus.append(raw_structure_dict[name])
            else:
                corpus.append("")  # 缺失填空

        # 返回密集矩阵
        X = self.vectorizer.fit_transform(corpus).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"N-gram Feature Dim: {X.shape[1]}")
        return X


# ==========================================
# 2. Loss Function
# ==========================================
class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(PearsonMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_cos = 1 - F.cosine_similarity(pred, target, dim=1).mean()
        return (1 - self.alpha) * loss_mse + self.alpha * loss_cos


# ==========================================
# 3. 双塔混合模型 (Hybrid Model)
# ==========================================
class HybridTwoTowerModel(nn.Module):
    def __init__(self, node_dim, num_relations, ngram_input_dim, output_dim, embedding_dim=16):
        super(HybridTwoTowerModel, self).__init__()

        # --------------------------------------------
        # 塔 A: Graph Tower (RGCN) - 负责拓扑结构
        # --------------------------------------------
        self.gnn_hidden = 64
        self.conv1 = RGCNConv(node_dim, self.gnn_hidden, num_relations)
        self.bn1 = nn.BatchNorm1d(self.gnn_hidden)

        self.conv2 = RGCNConv(self.gnn_hidden, self.gnn_hidden, num_relations)
        self.bn2 = nn.BatchNorm1d(self.gnn_hidden)

        self.dropout = nn.Dropout(0.2)

        # Graph Projection (Pooling后维度是 64*2=128)
        self.graph_proj = nn.Sequential(
            nn.Linear(self.gnn_hidden * 2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )

        # --------------------------------------------
        # 塔 B: Sequence Tower (MLP) - 负责 N-gram
        # --------------------------------------------
        # 这里复刻你之前的 DNN 结构
        self.ngram_mlp = nn.Sequential(
            nn.Linear(ngram_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),  # 压缩到与 Graph 同维
            nn.LeakyReLU(0.2)
        )

        # --------------------------------------------
        # 融合层 (Fusion)
        # --------------------------------------------
        # 输入 = Graph(64) + Ngram(64) = 128
        self.fusion_dim = 64 + 64

        # 这里的输出层我们也做一个 Embedding 提取，方便做 RSA 分析
        self.encoder_final = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, embedding_dim)  # 最终 Embedding (16维)
        )

        # 最后的回归预测头
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, data):
        # 1. 解包数据
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        ngram_x = data.ngram_feat  # 这里的维度是 [Batch, 1, NgramDim]

        # 调整 N-gram 维度 [Batch, 1, Dim] -> [Batch, Dim]
        if ngram_x.dim() > 2:
            ngram_x = ngram_x.squeeze(1)

        # ---------------------------
        # 2. 运行 Graph Tower
        # ---------------------------
        g = self.conv1(x, edge_index, edge_type)
        g = self.bn1(g)
        g = F.relu(g)
        g = self.dropout(g)

        g = self.conv2(g, edge_index, edge_type)
        g = self.bn2(g)
        g = F.relu(g)

        # 混合池化 (Mean + Max) 捕捉全面信息
        g_mean = global_mean_pool(g, batch)
        g_max = global_max_pool(g, batch)
        g_vec = torch.cat([g_mean, g_max], dim=1)  # (Batch, 128)

        g_feat = self.graph_proj(g_vec)  # (Batch, 64)

        # ---------------------------
        # 3. 运行 N-gram Tower
        # ---------------------------
        n_feat = self.ngram_mlp(ngram_x)  # (Batch, 64)

        # ---------------------------
        # 4. 融合 (Late Fusion)
        # ---------------------------
        # 拼接
        combined = torch.cat([g_feat, n_feat], dim=1)  # (Batch, 128)

        # 生成最终 Embedding
        emb = self.encoder_final(combined)

        # 预测分数
        pred = self.decoder(emb)

        return pred, emb


# ==========================================
# 4. 数据解析 (关键：同时注入 N-gram)
# ==========================================
def parse_litmus_json_hybrid(json_path, litmus_dict, ngram_matrix):
    """
    读取 JSON 图数据，并将对应的 N-gram 向量注入到 Data 对象中
    """
    import scipy.sparse

    # 1. 准备 N-gram Tensor
    if scipy.sparse.issparse(ngram_matrix):
        ngram_matrix = ngram_matrix.toarray()
    # 转为 Tensor
    ngram_tensor = torch.tensor(ngram_matrix, dtype=torch.float)

    # 2. 读取 JSON 内容
    raw_graphs = []
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            while pos < len(content) and content[pos].isspace(): pos += 1
            if pos >= len(content): break
            try:
                obj, end_pos = decoder.raw_decode(content[pos:])
                if isinstance(obj, list):
                    raw_graphs.extend(obj)
                else:
                    raw_graphs.append(obj)
                pos += end_pos
            except:
                break

    # 3. 对齐数据
    max_edge_type = 0
    aligned_graphs = [None] * len(litmus_dict)

    cnt = 0
    for item in raw_graphs:
        name = item['name']
        if name not in litmus_dict: continue
        idx = litmus_dict[name]

        # Node Features
        x = torch.tensor(item['node_features'], dtype=torch.float)

        # Edges
        raw_edges = torch.tensor(item['edges'], dtype=torch.long)
        if raw_edges.numel() > 0:
            edge_index = raw_edges[:, :2].t().contiguous()
            edge_type = raw_edges[:, 2]
            max_edge_type = max(max_edge_type, edge_type.max().item())
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        # === 注入 N-gram ===
        # 获取该 idx 对应的 N-gram 向量，增加维度方便 Batch
        ngram_vec = ngram_tensor[idx].unsqueeze(0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            ngram_feat=ngram_vec  # 保存 N-gram
        )
        aligned_graphs[idx] = data
        cnt += 1

    # 过滤 None
    valid_graphs = [g for g in aligned_graphs if g is not None]

    print(f"Hybrid Data Parsed: {len(valid_graphs)} graphs. Max edge type: {max_edge_type}")
    return valid_graphs, max_edge_type + 1


# ==========================================
# 5. 训练函数
# ==========================================
def train_hybrid_model(graph_data_list, target_performances, num_relations, ngram_dim, embedding_dim=16, epochs=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Label 处理
    target_log = np.log1p(target_performances)
    y_tensor = torch.FloatTensor(target_log)

    # 绑定 Label
    for i, data in enumerate(graph_data_list):
        data.y = y_tensor[i].unsqueeze(0)

    loader = GeoDataLoader(graph_data_list, batch_size=64, shuffle=True)

    # 初始化双塔模型
    node_dim = graph_data_list[0].x.shape[1]
    output_dim = target_performances.shape[1]

    model = HybridTwoTowerModel(node_dim, num_relations, ngram_dim, output_dim, embedding_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 双塔比较复杂，学习率小一点
    criterion = PearsonMSELoss(alpha=0.7).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    print(f"\n>>> 开始双塔模型训练 (Graph + Ngram)...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            preds, _ = model(batch)
            loss = criterion(preds, batch.y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.5f}")

    # 提取 Embedding
    print(">>> 训练完成，提取融合 Embedding...")
    model.eval()
    eval_loader = GeoDataLoader(graph_data_list, batch_size=64, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            _, emb = model(batch)
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# ==========================================
# 6. 评估函数
# ==========================================
def analyze_embedding_quality(embeddings, performances, method_name="My Embedding"):
    dist_perf_vector = pdist(performances, metric='correlation')
    dist_emb_vector = pdist(embeddings, metric='cosine')
    rho, p_val = spearmanr(dist_emb_vector, dist_perf_vector)
    print(f"=== {method_name} 评估结果 ===")
    print(f"Spearman Correlation (RSA Score): {rho:.4f}")
    if p_val < 0.05: print("结论: 显著相关")
    return rho


# ==========================================
# Main
# ==========================================
# 路径配置
graph_json_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"
litmus_cycle_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3.log"  # 存放原始文本的文件

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger(log_file=f"{stat_log_base}.check.run.log", level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Two-Tower Hybrid Evaluation | Seed={SEED} ===")

    # 1. 加载 Performance Data
    logger.info(f"Loading performance data: {cache_file_path}")
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
        logger.error("Cache file missing");
        exit(1)

    # 2. 建立 Index 映射
    param_dict, litmus_dict = {}, {}
    p_num, l_num = 0, 0
    for item in all_data:
        if str(item["param"]) not in param_dict:
            param_dict[str(item["param"])] = p_num
            p_num += 1
        if item["litmus"] not in litmus_dict:
            litmus_dict[item["litmus"]] = l_num
            l_num += 1

    # 构建 Score Matrix
    performance = np.zeros((l_num, p_num))
    for item in all_data:
        l_idx = litmus_dict[item["litmus"]]
        p_idx = param_dict[str(item["param"])]
        performance[l_idx][p_idx] = item["score"]

    # 3. 提取 N-gram 特征 (塔A 输入)
    logger.info("Extracting N-gram features...")
    if os.path.exists(litmus_cycle_path):
        with open(litmus_cycle_path, "r") as f:
            raw_lines = f.readlines()

        feature_extractor = LitmusFeatureExtractor(ngram_range=(1, 3))
        structure_map = feature_extractor.parse_raw_lines(raw_lines)

        # 按照 litmus_dict 的顺序生成矩阵
        ordered_names = [None] * l_num
        for name, idx in litmus_dict.items():
            ordered_names[idx] = name

        X_ngram = feature_extractor.fit_transform(ordered_names, structure_map)
    else:
        logger.error("Litmus text file missing!");
        exit(1)

    # 4. 解析图数据并融合 (塔B 输入 + 注入)
    logger.info("Parsing Graphs & Injecting N-grams...")
    if os.path.exists(graph_json_path):
        hybrid_graphs, num_relations = parse_litmus_json_hybrid(graph_json_path, litmus_dict, X_ngram)
    else:
        logger.error("Graph JSON missing!");
        exit(1)

    # 5. 训练双塔模型
    learned_emb = train_hybrid_model(
        graph_data_list=hybrid_graphs,
        target_performances=performance,
        num_relations=num_relations,
        ngram_dim=X_ngram.shape[1],  # N-gram 维度自动获取
        embedding_dim=16,
        epochs=500
    )

    # 6. 评估
    print("\n------------------------------------------------")
    print("【双塔混合模型】Graph(RGCN) + Sequence(MLP) 融合嵌入质量：")
    analyze_embedding_quality(learned_emb, performance, method_name="Hybrid Two-Tower")

    # 7. 保存
    output_emb_dict = {}
    for i, name in enumerate(ordered_names):
        if name: output_emb_dict[name] = learned_emb[i].tolist()

    with open("learned_hybrid_embeddings.json", "w") as f:
        json.dump(output_emb_dict, f, indent=4)
    logger.info("Saved hybrid embeddings.")