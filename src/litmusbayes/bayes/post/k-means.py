import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

# ================= 配置 =================
# 向量文件路径
LITMUS_VEC_PATHS = [
    "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log",
]

# 【新增】基准分数文件路径
# 请确保这个文件里包含你提供的那些 "TestName": Score 数据
# 如果是标准的 JSON 文件（带大括号 {}），代码能读；
# 如果是每行类似 "ISA09": 11321.4, 的格式，代码也能读。
BASELINE_SCORES_PATH = "../log/baseline_scores_0.json"

OUTPUT_DIR = "./cluster_results"
SEED = 2025


# ================= 数据加载类 =================

def load_baseline_whitelist(path):
    """
    读取 baseline_scores 文件，返回由 Litmus 名称组成的集合(Set)
    """
    if not os.path.exists(path):
        print(f"Error: Baseline file not found at {path}")
        return set()

    valid_keys = set()
    print(f"Loading baseline whitelist from {path}...")

    try:
        # 尝试 1: 当作标准 JSON 读取
        with open(path, 'r') as f:
            data = json.load(f)
            valid_keys = set(data.keys())
    except json.JSONDecodeError:
        # 尝试 2: 如果不是标准 JSON (例如没有外层 {} 或每行一个 Key:Value)，则按行解析
        print("File is not standard JSON, parsing line by line...")
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                # 去掉行末逗号
                if line.endswith(','): line = line[:-1]
                if ":" in line:
                    # 提取冒号前的部分作为 Key
                    key_part = line.split(":", 1)[0].strip()
                    # 去掉引号
                    key_part = key_part.strip('"').strip("'")
                    if key_part:
                        valid_keys.add(key_part)

    print(f"Found {len(valid_keys)} valid litmus tests in baseline.")
    return valid_keys


class LitmusVectorLoader:
    def __init__(self, paths):
        self.paths = paths
        self.litmus_to_vec = {}

    def load(self, whitelist=None):
        """
        :param whitelist: 一个集合(set)，如果提供了，只保留在这个集合里的 litmus
        """
        print("Loading vectors...")
        for path in self.paths:
            if not os.path.exists(path):
                print(f"Warning: Path not found: {path}")
                continue
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line: continue
                    try:
                        name, vec_str = line.split(":", 1)
                        name = name.strip()

                        # 【过滤逻辑】如果提供了白名单，且该文件不在白名单里，直接跳过
                        if whitelist is not None and name not in whitelist:
                            continue

                        vec = eval(vec_str)
                        self.litmus_to_vec[name] = list(vec)
                    except Exception as e:
                        # 忽略解析错误
                        pass

        print(f"Loaded {len(self.litmus_to_vec)} vectors that matched baseline.")
        return self.litmus_to_vec


# ================= 聚类分析核心类 =================
class LitmusClusterAnalysis:
    def __init__(self, data_dict):
        self.names = list(data_dict.keys())
        self.X = np.array(list(data_dict.values()))

        # 数据标准化
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        self.kmeans_model = None
        self.labels = None

    def determine_optimal_k(self, max_k=20):
        print(f"Calculating inertia for K=1 to {max_k}...")
        inertias = []
        K_range = range(1, max_k + 1)

        for k in tqdm(K_range):
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            km.fit(self.X_scaled)
            inertias.append(km.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        save_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")
        plt.savefig(save_path)
        print(f"Elbow plot saved to {save_path}")

    def run_clustering(self, n_clusters):
        print(f"Running K-Means with K={n_clusters}...")
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        self.labels = self.kmeans_model.fit_predict(self.X_scaled)

        centers = self.kmeans_model.cluster_centers_
        closest_indices, _ = pairwise_distances_argmin_min(centers, self.X_scaled)

        results = {}
        for cluster_id in range(n_clusters):
            members_indices = np.where(self.labels == cluster_id)[0]
            members = [self.names[i] for i in members_indices]

            center_idx = closest_indices[cluster_id]
            representative = self.names[center_idx]

            results[cluster_id] = {
                "representative": representative,
                "count": len(members),
                "members": members
            }
        return results

    def visualize_tsne(self, output_file="tsne_cluster.png"):
        print("Running t-SNE for visualization...")
        # 【修复】删除了 n_iter=1000 以兼容 sklearn 版本
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
        X_embedded = tsne.fit_transform(self.X_scaled)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            hue=self.labels,
            palette="viridis",
            s=60,
            alpha=0.7,
            legend="full"
        )

        closest_indices, _ = pairwise_distances_argmin_min(self.kmeans_model.cluster_centers_, self.X_scaled)
        plt.scatter(
            X_embedded[closest_indices, 0],
            X_embedded[closest_indices, 1],
            c='red', s=200, marker='*', label='Centroids'
        )

        plt.title(f"t-SNE Visualization (Filtered by Baseline, K={self.kmeans_model.n_clusters})")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, output_file))
        print(f"Visualization saved to {os.path.join(OUTPUT_DIR, output_file)}")


# ================= 主流程 =================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 先加载 Baseline 白名单
    whitelist = load_baseline_whitelist(BASELINE_SCORES_PATH)

    if len(whitelist) == 0:
        print("Warning: Whitelist is empty! Check your baseline file path and format.")
        # 询问用户是否继续（如果为空可能导致后续报错，最好直接退出）
        print("Exiting to prevent empty clustering.")
        exit(1)

    # 2. 加载向量 (带过滤)
    loader = LitmusVectorLoader(LITMUS_VEC_PATHS)
    # 将 whitelist 传进去，不在名单里的向量直接丢弃
    data = loader.load(whitelist=whitelist)

    if not data:
        print("No valid data loaded after filtering. Exiting.")
        exit()

    print(f"Final dataset size for clustering: {len(data)}")

    analyzer = LitmusClusterAnalysis(data)

    # 3. 聚类 (这里设为10，你可以根据 elbows 图修改)
    # analyzer.determine_optimal_k(max_k=30) # 第一次跑可以解开这行看K值

    CHOSEN_K = 10
    clusters = analyzer.run_clustering(n_clusters=CHOSEN_K)

    # 4. 保存结果
    output_json = os.path.join(OUTPUT_DIR, "cluster_centers.json")

    print("\n=== Clustering Representatives (Filtered) ===")
    for cid, info in clusters.items():
        print(f"Cluster {cid}: {info['count']} tests. \t Representative: {info['representative']}")

    with open(output_json, "w") as f:
        json.dump(clusters, f, indent=4)
    print(f"\nDetailed cluster info saved to {output_json}")

    # 5. 可视化
    analyzer.visualize_tsne()