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
from sklearn.model_selection import train_test_split

# --- 新增 PyG 依赖 ---
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, RGCNConv
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
class RGCNLitmusEmbedder(nn.Module):
    def __init__(self, node_feat_dim, output_dim, num_relations, embedding_dim=16):
        """
        num_relations: 边的类型总数 (例如如果类型是 0-7，则 num_relations=8)
        """
        super(RGCNLitmusEmbedder, self).__init__()

        # --- RGCN 层 ---
        # RGCN 需要知道有多少种关系，以便为每种关系分配不同的权重
        self.conv1 = RGCNConv(node_feat_dim, 64, num_relations)
        self.conv2 = RGCNConv(64, 128, num_relations)

        # RGCN 参数量较大，通常两层就够了，或者加 Dropout
        self.dropout = nn.Dropout(0.2)

        # --- 降维/嵌入层 ---
        self.fc_emb = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, embedding_dim)
        )

        # --- 解码器 ---
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, data):
        # 注意：RGCN 的 forward 需要传入 edge_type
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        # 1. Message Passing (RGCN)
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)

        # 2. Pooling
        x = global_mean_pool(x, batch) + global_max_pool(x, batch)

        # 3. Generate Embedding
        emb = self.fc_emb(x)

        # 4. Predict
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

    model = RGCNLitmusEmbedder(node_dim, output_dim, num_relations, embedding_dim=16).to(device)
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

    return np.vstack(all_embeddings), model


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
# 6. 新增：新数据推理逻辑 (专门用于新文件)
# ==========================================
def parse_new_json_for_inference(json_path):
    """
    专门解析用于推理的新 JSON 文件。
    不需要 litmus_name_to_idx 字典，直接读取所有图数据。
    """
    raw_graphs = []
    # 读取文件
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

    parsed_graphs = []
    parsed_names = []

    for item in raw_graphs:
        name = item['name']

        # 1. Node Features
        x = torch.tensor(item['node_features'], dtype=torch.float)

        # 2. Edges & Types
        raw_edges = torch.tensor(item['edges'], dtype=torch.long)

        if raw_edges.numel() > 0:
            edge_index = raw_edges[:, :2].t().contiguous()
            edge_type = raw_edges[:, 2]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        parsed_graphs.append(data)
        parsed_names.append(name)

    print(f"Parsed {len(parsed_graphs)} new graphs for inference.")
    return parsed_graphs, parsed_names


def generate_embeddings_for_new_graph_files(trained_model, new_json_path, output_file_path):
    """
    读取新 Graph JSON -> 推理 -> 保存
    """
    print(f"\n>>> 开始处理新 Graph 数据推理...")
    print(f"Input: {new_json_path}")

    if not os.path.exists(new_json_path):
        print(f"Error: File not found {new_json_path}")
        return

    # 1. 解析数据
    new_graphs, new_names = parse_new_json_for_inference(new_json_path)

    if not new_graphs:
        print("No valid graphs found.")
        return

    # 2. 准备 Loader
    # 这里的 batch_size 可以设大一点，加快推理
    loader = GeoDataLoader(new_graphs, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()

    all_embeddings = []
    print("Running GNN inference...")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # GNN 的 forward 只需要 batch 数据，不需要 y
            _, emb = trained_model(batch)
            all_embeddings.append(emb.cpu().numpy())

    final_embeddings = np.vstack(all_embeddings)

    # 3. 保存
    print(f"Saving results to {output_file_path} ...")
    with open(output_file_path, "w") as f_out:
        for i, name in enumerate(new_names):
            vec = final_embeddings[i].tolist()
            # 格式: Name:[v1, v2, ...]
            vec_str = json.dumps(vec)
            f_out.write(f"{name}:{vec_str}\n")

    print("Done.")



# ==========================================
# Main Process
# ==========================================
# 配置路径 (请修改为你包含 node_features 的 json 文件路径)
graph_json_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"
# 其他路径保持不变...
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"


inference_graph_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"  # 请修改你的新文件路径
inference_output_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_rgcn.log"
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logger = setup_logger(log_file=f"{stat_log_base}.check.run.log", level=logging.INFO, name=LOG_NAME)

    # 1. 加载所有性能数据并识别唯一的 Litmus 名字
    all_raw_data = []
    unique_names = set()
    with open(cache_file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            all_raw_data.append(item)
            unique_names.add(item["litmus"])

    all_litmus_list = sorted(list(unique_names))

    # 2. 【关键】划分训练集和测试集名字
    train_names, test_names = train_test_split(all_litmus_list, test_size=0.2, random_state=SEED)
    logger.info(f"Split data: Train={len(train_names)}, Test={len(test_names)}")

    # 3. 构建索引映射和全量 Performance 矩阵
    all_params = sorted(list(set([str(i["param"]) for i in all_raw_data])))
    param_dict = {p: i for i, p in enumerate(all_params)}
    litmus_dict = {name: i for i, name in enumerate(all_litmus_list)}

    full_perf = np.zeros((len(all_litmus_list), len(all_params)))
    for item in all_raw_data:
        full_perf[litmus_dict[item["litmus"]]][param_dict[str(item["param"])]] = item["score"]

    # 4. 解析全量图数据
    all_graphs, num_relations = parse_litmus_json_to_pyg(graph_json_path, litmus_dict)

    # 5. 根据划分的名字提取对应的 Graph 和 Performance
    # for test_name in test_names:
    #     print(test_name)
    train_idx = [litmus_dict[n] for n in train_names]
    test_idx = [litmus_dict[n] for n in test_names]

    train_graphs = [all_graphs[i] for i in train_idx]
    test_graphs = [all_graphs[i] for i in test_idx]
    train_perf = full_perf[train_idx]
    test_perf = full_perf[test_idx]

    # 6. 训练与推理
    train_embs, model = train_gnn_and_get_embeddings(
        graph_data_list=train_graphs,
        target_performances=train_perf,
        num_relations=num_relations,  # 传入边类型数量
        embedding_dim=16,
        epochs=500
    )


    def get_any_emb(model, data_list, device):
        model.eval()
        loader = GeoDataLoader(data_list, batch_size=32, shuffle=False)
        all_e = []
        with torch.no_grad():
            for b in loader:
                _, e = model(b.to(device))
                all_e.append(e.cpu().numpy())
        return np.vstack(all_e)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_embs = get_any_emb(model, test_graphs, device)

    # 7. 评估
    print("\n" + "=" * 30)
    analyze_embedding_quality(train_embs, train_perf, "Train Set (Seen)")
    analyze_embedding_quality(test_embs, test_perf, "Test Set (Unseen)")
    print("=" * 30)

    generate_embeddings_for_new_graph_files(
        model,
        inference_graph_path,
        inference_output_file
    )