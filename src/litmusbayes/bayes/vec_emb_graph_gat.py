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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- 新增 PyG 依赖 ---
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATv2Conv, global_max_pool, GlobalAttention
# 保持原有的工具引用
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ==========================================
# 1. 自定义 Loss (保持不变)
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
# 2. 新增：GNN 模型定义
# ==========================================
class GATLitmusEmbedder(nn.Module):
    def __init__(self, node_feat_dim, output_dim, num_relations, embedding_dim=16):
        super(GATLitmusEmbedder, self).__init__()

        # 1. 边特征嵌入
        # 我们把 edge_type (0,1,2...) 变成一个向量，喂给 GAT
        self.edge_emb = nn.Embedding(num_relations, 16)

        # 2. GATv2 层 (带注意力机制)
        # edge_dim=16: 告诉 GAT 我们有边特征
        # heads=4: 多头注意力，捕捉不同的依赖模式
        self.conv1 = GATv2Conv(node_feat_dim, 64, heads=4, concat=True, edge_dim=16)
        # out: 64 * 4 = 256

        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = GATv2Conv(256, 128, heads=2, concat=True, edge_dim=16)
        # out: 128 * 2 = 256

        self.bn2 = nn.BatchNorm1d(256)

        # 3. 关键修改：Global Attention Pooling (或者只用 Max)
        # 这种 Pooling 允许模型给每个节点打分，只聚合分高的节点
        # 类似于 N-gram 里的 "Key Token"
        self.pool_gate = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.pool = GlobalAttention(gate_nn=self.pool_gate)

        # 4. 降维
        self.fc_emb = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, embedding_dim)
        )

        # 5. 预测
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        # 生成边特征
        edge_attr = self.edge_emb(edge_type)

        # --- GNN Layers ---
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        # --- Pooling (关键一步) ---
        # 方案 A: Global Attention Pooling (推荐，自动寻找关键节点)
        x = self.pool(x, batch)

        # 方案 B: 纯 Max Pooling (如果你觉得 Attention 不稳定，用这个)
        # x = global_max_pool(x, batch)

        # --- Output ---
        emb = self.fc_emb(x)
        pred = self.decoder(emb)

        return pred, emb


# ==========================================
# 3. 新增：数据加载与解析逻辑
# ==========================================
def parse_litmus_json_to_pyg(json_path, litmus_name_to_idx):
    """
    读取你的 JSON 文件，并转换为 PyG 的 Data 对象列表
    需要 litmus_name_to_idx 来确保顺序与 Performance 矩阵对齐
    """
    # 读取所有 JSON 数据
    raw_graphs = []
    # 支持流式读取多行 JSON 数组
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

    # 记录最大的边类型 ID，以便初始化模型
    max_edge_type = 0

    aligned_graphs = [None] * len(litmus_name_to_idx)
    cnt = 0

    for item in raw_graphs:
        name = item['name']
        if name not in litmus_name_to_idx:
            continue
        idx = litmus_name_to_idx[name]

        # 1. Node Features
        x = torch.tensor(item['node_features'], dtype=torch.float)

        # 2. Edges & Types
        raw_edges = torch.tensor(item['edges'], dtype=torch.long)

        if raw_edges.numel() > 0:  # 确保有边
            # 前两列是连接关系 [Source, Target]
            edge_index = raw_edges[:, :2].t().contiguous()
            # 第三列是类型 [Type]
            edge_type = raw_edges[:, 2]

            # 更新最大类型ID
            current_max = edge_type.max().item()
            if current_max > max_edge_type:
                max_edge_type = current_max
        else:
            # 处理没有边的孤立图情况
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        # --- 创建 Data 对象，必须包含 edge_type ---
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        aligned_graphs[idx] = data
        cnt += 1

    # 检查是否有缺失
    valid_graphs = []
    valid_indices = []  # 记录哪些行是有效的
    for i, g in enumerate(aligned_graphs):
        if g is not None:
            valid_graphs.append(g)
            valid_indices.append(i)
        else:
            # 如果缺失，可以用一个全0的虚拟图填充，或者在后续训练剔除
            # 这里简单起见，我们假设数据是齐全的，或者你需要处理缺失
            print(f"Warning: Index {i} missing graph data.")
            # 创建个 dummy 节点防止报错 (1个节点, 全0特征)
            dummy_x = torch.zeros((1, len(raw_graphs[0]['node_features'][0])))
            dummy_edge = torch.tensor([[0], [0]], dtype=torch.long)
            valid_graphs.append(Data(x=dummy_x, edge_index=dummy_edge))

    print(f"Converted {cnt} graphs. Max edge type ID found: {max_edge_type}")
    return valid_graphs, max_edge_type + 1


# ==========================================
# 4. 修改：训练函数 (适配 PyG DataLoader)
# ==========================================
def train_gnn_and_get_embeddings(graph_data_list, target_performances, num_relations, embedding_dim=16, epochs=500):
    """
    graph_data_list: List[Data]
    target_performances: (N, 70) numpy array
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 准备 Target
    target_log = np.log1p(target_performances)
    y_tensor = torch.FloatTensor(target_log)

    # 2. 将 Target 绑定到 Data 对象中以便 Batch 分发
    for i, data in enumerate(graph_data_list):
        data.y = y_tensor[i].unsqueeze(0)  # (1, 70)

    # 3. 创建 DataLoader
    loader = GeoDataLoader(graph_data_list, batch_size=64, shuffle=True)

    # 4. 初始化模型
    # 获取节点特征维度
    node_dim = graph_data_list[0].x.shape[1]
    output_dim = target_performances.shape[1]

    model = GATLitmusEmbedder(node_dim, output_dim, num_relations, embedding_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = PearsonMSELoss(alpha=0.7).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    print(f"\n>>> 开始 GNN 训练 (NodeDim:{node_dim} -> Emb:{embedding_dim}) Device:{device}")

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

    # 5. 提取嵌入 (Full Batch Inference)
    print(">>> 训练完成，正在提取图嵌入...")
    model.eval()

    # 为了保持顺序，创建一个不打乱的 Loader
    eval_loader = GeoDataLoader(graph_data_list, batch_size=64, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            _, emb = model(batch)
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


# ==========================================
# 5. 评估工具 (保持不变)
# ==========================================
def analyze_embedding_quality(embeddings, performances, method_name="My Embedding"):
    # ... (保持你原有的评估代码不变) ...
    dist_perf_vector = pdist(performances, metric='correlation')
    dist_emb_vector = pdist(embeddings, metric='cosine')
    rho, p_val = spearmanr(dist_emb_vector, dist_perf_vector)

    print(f"=== {method_name} 评估结果 ===")
    print(f"Spearman Correlation (RSA Score): {rho:.4f}")
    if p_val < 0.05: print("结论: 显著相关")

    # 如果想画图，解除下面的注释
    # ... (你的画图代码)
    return rho


# ==========================================
# Main Process
# ==========================================
# 配置路径 (请修改为你包含 node_features 的 json 文件路径)
graph_json_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"
# 其他路径保持不变...
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"

if __name__ == "__main__":
    # 0. 基础设置
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 初始化日志
    logger = setup_logger(log_file=f"{stat_log_base}.check.run.log", level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start RGCN Evaluation Run | Seed={SEED} ===")

    # ==========================================
    # 1. 加载 Performance 数据 (构建 Ground Truth)
    # ==========================================
    logger.info(f"Loading performance data from {cache_file_path} ...")
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
        logger.error(f"Cache file not found: {cache_file_path}")
        exit(1)

    logger.info(f"Loaded {len(all_data)} performance records.")

    # ==========================================
    # 2. 构建 ID 映射 (Name -> Index)
    # ==========================================
    # 我们需要先知道一共有多少个 Litmus Test，才能去解析图数据并对齐
    param_dict = {}
    param_num = 0
    litmus_dict = {}
    litmus_num = 0

    for item in all_data:
        # 建立参数映射
        p_str = str(item["param"])
        if p_str not in param_dict:
            param_dict[p_str] = param_num
            param_num += 1

        # 建立 Litmus 名字映射
        l_name = item["litmus"]
        if l_name not in litmus_dict:
            litmus_dict[l_name] = litmus_num
            litmus_num += 1

    logger.info(f"Unique Litmus Tests: {litmus_num}")
    logger.info(f"Unique Parameter Configs: {param_num}")

    # ==========================================
    # 3. 构建 Performance 矩阵 (N x M)
    # ==========================================
    performance = np.zeros((litmus_num, param_num))
    for item in all_data:
        l_idx = litmus_dict[item["litmus"]]
        p_idx = param_dict[str(item["param"])]
        performance[l_idx][p_idx] = item["score"]

    # ==========================================
    # 4. 解析图数据 (RGCN Input)
    # ==========================================
    logger.info("Parsing Graph JSON with Edge Types...")

    if os.path.exists(graph_json_path):
        # 关键：传入 litmus_dict，确保返回的图列表顺序和 performance 矩阵一致
        aligned_graphs, num_relations = parse_litmus_json_to_pyg(graph_json_path, litmus_dict)
        logger.info(f"Graph parsing done. Detected {num_relations} distinct edge types.")
    else:
        logger.error(f"Graph JSON file not found: {graph_json_path}")
        exit(1)

    # ==========================================
    # 5. 训练 RGCN 模型
    # ==========================================
    logger.info(f">>> Training RGCN to learn embeddings (Num Relations: {num_relations})...")

    # 这里的参数顺序必须和函数定义一致：
    # graphs, targets, num_relations, embedding_dim, epochs
    learned_emb = train_gnn_and_get_embeddings(
        graph_data_list=aligned_graphs,
        target_performances=performance,
        num_relations=num_relations,  # 传入边类型数量
        embedding_dim=16,
        epochs=500
    )

    # ==========================================
    # 6. 评估结果
    # ==========================================
    print("\n------------------------------------------------")
    print("【GNN (GAT) 模型】基于图结构学习到的嵌入 (16维) 的质量：")
    analyze_embedding_quality(learned_emb, performance, method_name="RGCN Embedding")

    # ==========================================
    # 7. 保存结果
    # ==========================================
    # 反转字典以便通过 ID 查名字
    ordered_litmus_names = [None] * litmus_num
    for name, idx in litmus_dict.items():
        ordered_litmus_names[idx] = name

    output_emb_dict = {}
    for i, name in enumerate(ordered_litmus_names):
        # 只有当名字存在且对应的图数据也存在时才保存(虽然后者通常是对齐的)
        if name:
            output_emb_dict[name] = learned_emb[i].tolist()

    out_file = "learned_gnn_embeddings.json"
    with open(out_file, "w") as f:
        json.dump(output_emb_dict, f, indent=4)
    logger.info(f"RGCN embeddings saved to {out_file}")