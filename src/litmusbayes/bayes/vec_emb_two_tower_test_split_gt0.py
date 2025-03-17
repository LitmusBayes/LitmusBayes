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
            token_pattern=r'(?u)\b\w[\w.]*\b',
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
        corpus_segmented = []
        for name in litmus_name_list:
            if name in raw_structure_dict:
                # 1. 先按 SEP 切分成多个线程的指令序列
                threads = raw_structure_dict[name].split("SEP")
                # 2. 将每个线程作为独立的文档加入语料库
                corpus_segmented.extend(threads)
            else:
                corpus_segmented.append("")

        # 3. 训练词库（此时 vectorizer 永远看不到跨线程的组合）
        self.vectorizer.fit(corpus_segmented)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # 4. 转换时，需要把同一个 litmus 程序的多个线程特征累加起来
        final_X = []
        for name in litmus_name_list:
            if name in raw_structure_dict:
                threads = raw_structure_dict[name].split("SEP")
                # 提取该程序下所有线程的特征并求和
                thread_vectors = self.vectorizer.transform(threads).toarray()
                # sample_vector = thread_vectors.sum(axis=0)
                sample_vector = np.prod(thread_vectors + 1, axis=0)
                final_X.append(sample_vector)
            else:
                final_X.append(np.zeros(len(self.feature_names)))
        print(np.array(final_X).shape)
        return np.array(final_X)


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

    return np.vstack(all_embeddings), model


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
graph_json_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2_gt0.jsonl"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_gt_0_for_graph.jsonl"
litmus_cycle_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3_gt0.log"  # 存放原始文本的文件

# ==========================================
# 完整集成版 Main
# ==========================================
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 基础配置
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger(log_file=f"{stat_log_base}.check.run.log", level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Two-Tower Hybrid Evaluation (Train/Test Split) | Seed={SEED} ===")

    # 1. 加载 Performance Data
    logger.info(f"Loading performance data: {cache_file_path}")
    all_raw_data = []
    unique_litmus_names = set()
    unique_params = sorted(list(set()))  # 用于保存参数列

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_raw_data.append(item)
                    unique_litmus_names.add(item["litmus"])
    else:
        logger.error("Cache file missing!")
        exit(1)

    # 2. 【核心修改】划分训练集和测试集 (按 Litmus 程序名划分)
    all_litmus_list = sorted(list(unique_litmus_names))
    train_names, test_names = train_test_split(
        all_litmus_list,
        test_size=0.20,
        random_state=SEED
    )

    logger.info(f"Total Litmus Programs: {len(all_litmus_list)}")
    logger.info(f"Training set: {len(train_names)} | Test set: {len(test_names)}")

    # 3. 建立统一索引映射
    # 获取所有唯一的参数并排序，保证矩阵列定义一致
    all_params_str = sorted(list(set([str(item["param"]) for item in all_raw_data])))
    param_dict = {p_str: i for i, p_str in enumerate(all_params_str)}
    litmus_dict = {name: i for i, name in enumerate(all_litmus_list)}

    # 构建完整的 Performance 矩阵
    full_performance = np.zeros((len(all_litmus_list), len(all_params_str)))
    for item in all_raw_data:
        l_idx = litmus_dict[item["litmus"]]
        p_idx = param_dict[str(item["param"])]
        full_performance[l_idx][p_idx] = item["score"]

    # 4. 提取 N-gram 特征 (全量提取)
    logger.info("Extracting N-gram features for all programs...")
    if os.path.exists(litmus_cycle_path):
        with open(litmus_cycle_path, "r") as f:
            raw_lines = f.readlines()
        feature_extractor = LitmusFeatureExtractor(ngram_range=(1, 3))
        structure_map = feature_extractor.parse_raw_lines(raw_lines)
        X_ngram_all = feature_extractor.fit_transform(all_litmus_list, structure_map)
    else:
        logger.error("Litmus text file missing!");
        exit(1)

    # 5. 解析图数据 (全量解析)
    logger.info("Parsing Graphs & Injecting N-grams...")
    all_hybrid_graphs, num_relations = parse_litmus_json_hybrid(graph_json_path, litmus_dict, X_ngram_all)

    # 6. 【数据分割】将 Graph 和 Performance 分成 Train/Test 两份
    train_indices = [litmus_dict[name] for name in train_names]
    test_indices = [litmus_dict[name] for name in test_names]

    train_graphs = [all_hybrid_graphs[i] for i in train_indices]
    test_graphs = [all_hybrid_graphs[i] for i in test_indices]

    train_perf = full_performance[train_indices]
    test_perf = full_performance[test_indices]

    # 7. 训练模型 (仅使用训练集)
    # 注意：这里的 target_performances 传入的是 train_perf
    learned_train_emb, trained_model = train_hybrid_model(
        graph_data_list=train_graphs,
        target_performances=train_perf,
        num_relations=num_relations,
        ngram_dim=X_ngram_all.shape[1],
        embedding_dim=16,
        epochs=500
    )

    # 8. 推理测试集 Embedding
    logger.info("Inference: Generating embeddings for test set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 重新加载模型架构来提取测试集（train_hybrid_model 内部会创建模型，这里我们手动获取）
    # 为了简单起见，我们可以在 train_hybrid_model 结束后返回模型对象，
    # 或者在这里直接用最后一次训练好的状态进行一次 eval。

    # 技巧：由于 train_hybrid_model 内部没有返回 model，我们直接对测试集运行一次 DataLoader
    # (假设你已将 train_hybrid_model 修改为返回 (embeddings, model))
    # 如果没改，最简单的方法是在训练函数里最后顺便把 test_graphs 的 emb 也跑出来。

    # 这里演示如何直接获取测试集 Emb (通常建议修改 train_hybrid_model 返回 model)
    # 暂且为了代码完整性，我们重新跑一下 eval 逻辑:
    def get_any_emb(model, data_list, device):
        model.eval()
        loader = GeoDataLoader(data_list, batch_size=32, shuffle=False)
        all_e = []
        with torch.no_grad():
            for b in loader:
                _, e = model(b.to(device))
                all_e.append(e.cpu().numpy())
        return np.vstack(all_e)


    # ---------------------------------------------------------
    # 注意：为了让下面代码跑通，你需要微调一下 train_hybrid_model
    # 让它最后一行 return np.vstack(all_embeddings), model
    # ---------------------------------------------------------
    # 这里假设你已经按上述建议修改了返回值：
    # learned_train_emb, trained_model = train_hybrid_model(...)

    # 如果不想改 train 函数，就在 train 函数内部把测试集逻辑加进去，如下：
    learned_test_emb = get_any_emb(trained_model, test_graphs, device)
    # (此处建议在 train_hybrid_model 结尾处，对 test_graphs 同样做一次 model(batch) 并返回)

    # 9. 最终评估
    print("\n" + "=" * 60)
    print("【双塔混合模型评估报告】")
    print(f"训练集规模: {len(train_names)} | 测试集规模: {len(test_names)}")
    print("-" * 60)

    # 评估训练集
    analyze_embedding_quality(learned_train_emb, train_perf, method_name="Train Set (Seen)")

    # 评估测试集 (需要你从训练函数拿回模型或在里面计算)
    analyze_embedding_quality(learned_test_emb, test_perf, method_name="Test Set (Unseen)")
    print("=" * 60)

    # 10. 保存测试集清单方便后续分析
    with open("test_set_litmus.json", "w") as f:
        json.dump(test_names, f, indent=4)
    logger.info("Workflow completed. Test set names saved.")