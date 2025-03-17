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
    # "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log",
    "../RQ3/vector/litmus_vector_two_tower_fixed.log",
]

# 基准分数文件路径
BASELINE_SCORES_PATH = "../../log/baseline_scores.json"

OUTPUT_DIR = "./cluster_results_final_all_15"
CHOSEN_K = 15
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
        with open(path, 'r') as f:
            data = json.load(f)
            valid_keys = set(data.keys())
    except json.JSONDecodeError:
        print("File is not standard JSON, parsing line by line...")
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith(','): line = line[:-1]
                if ":" in line:
                    key_part = line.split(":", 1)[0].strip()
                    key_part = key_part.strip('"').strip("'")
                    if key_part:
                        valid_keys.add(key_part)

    print(f"Found {len(valid_keys)} valid litmus tests in baseline.")
    return valid_keys


class LitmusVectorLoader:
    def __init__(self, paths):
        self.paths = paths

    def load(self, whitelist=None):
        """
        :param whitelist: 一个集合(set)，如果提供了，只保留在这个集合里的 litmus
        :return: 每次调用返回一个全新的字典，避免全量数据和过滤数据相互污染
        """
        litmus_to_vec = {}
        msg = "all tests" if whitelist is None else "baseline-filtered tests"
        print(f"Loading vectors for {msg}...")

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
                        litmus_to_vec[name] = list(vec)
                    except Exception as e:
                        pass

        print(f"Loaded {len(litmus_to_vec)} vectors.")
        return litmus_to_vec


# ================= 聚类分析核心类 =================
class LitmusClusterAnalysis:
    def __init__(self, data_dict):
        self.names = list(data_dict.keys())
        self.X = np.array(list(data_dict.values()))

        # 数据标准化：在过滤后的数据上拟合（fit）并转换（transform）
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

        results = {}
        for cluster_id in range(n_clusters):
            members_indices = np.where(self.labels == cluster_id)[0]
            center_vec = centers[cluster_id]
            cluster_points = self.X_scaled[members_indices]
            distances = np.linalg.norm(cluster_points - center_vec, axis=1)

            sorted_relative_idx = np.argsort(distances)
            sorted_members_indices = members_indices[sorted_relative_idx]
            sorted_members = [self.names[i] for i in sorted_members_indices]

            representative = sorted_members[0]

            results[cluster_id] = {
                "representative": representative,
                "count": len(sorted_members),
                "members": sorted_members
            }

        return results

    def assign_new_data(self, new_data_dict):
        """
        【新增方法】
        使用已经基于 baseline 数据计算出的聚类中心，对另一份数据（如全量数据）进行聚类分配
        """
        print("Assigning all data to previously computed cluster centers...")
        if self.kmeans_model is None:
            raise ValueError("Model is not trained. Please run `run_clustering` first.")

        new_names = list(new_data_dict.keys())
        new_X = np.array(list(new_data_dict.values()))

        # 必须使用之前过滤数据拟合好的 scaler 进行 transform (不能使用 fit_transform)
        new_X_scaled = self.scaler.transform(new_X)

        # 使用过滤数据训练好的 kmeans 模型对全量数据进行预测（分配所属簇）
        new_labels = self.kmeans_model.predict(new_X_scaled)
        centers = self.kmeans_model.cluster_centers_

        results_all = {}
        for cluster_id in range(self.kmeans_model.n_clusters):
            members_indices = np.where(new_labels == cluster_id)[0]
            center_vec = centers[cluster_id]

            # 以防全量数据中有簇未被分配到任何数据（理论上全量包含过滤，不会发生此情况）
            if len(members_indices) == 0:
                results_all[cluster_id] = {
                    "representative": None,
                    "count": 0,
                    "members": []
                }
                continue

            cluster_points = new_X_scaled[members_indices]
            distances = np.linalg.norm(cluster_points - center_vec, axis=1)

            sorted_relative_idx = np.argsort(distances)
            sorted_members_indices = members_indices[sorted_relative_idx]
            sorted_members = [new_names[i] for i in sorted_members_indices]

            representative = sorted_members[0]

            results_all[cluster_id] = {
                "representative": representative,
                "count": len(sorted_members),
                "members": sorted_members
            }

        return results_all

    def visualize_tsne(self, output_file="tsne_cluster_filtered.png"):
        print("Running t-SNE for visualization...")
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
        print("Exiting to prevent empty clustering.")
        exit(1)

    # 2. 实例化加载器并分别加载两份数据
    loader = LitmusVectorLoader(LITMUS_VEC_PATHS)

    # 2.1 加载过滤后的数据 (用于计算聚类中心)
    filtered_data = loader.load(whitelist=whitelist)

    # 2.2 【新增】加载不进行过滤的全量数据 (用于后续分配)
    all_data = loader.load(whitelist=None)

    if not filtered_data:
        print("No valid data loaded after filtering. Exiting.")
        exit()

    print(f"Filtered dataset size (for training K-Means): {len(filtered_data)}")
    print(f"Full dataset size (for assignment): {len(all_data)}")

    # 3. 使用【过滤后的数据】初始化分析类和训练 KMeans
    analyzer = LitmusClusterAnalysis(filtered_data)
    clusters_filtered = analyzer.run_clustering(n_clusters=CHOSEN_K)

    # 4. 【新增】使用前面训练好的模型，把【全量数据】映射到已有的聚类中心上
    clusters_all = analyzer.assign_new_data(all_data)

    # 5. 分别保存两份结果文件
    output_filtered_json = os.path.join(OUTPUT_DIR, "cluster_centers_filtered.json")
    output_all_json = os.path.join(OUTPUT_DIR, "cluster_centers_all.json")

    # 5.1 打印并保存过滤后的结果
    print("\n=== Clustering Representatives (Filtered Data) ===")
    for cid, info in clusters_filtered.items():
        print(f"Cluster {cid}: {info['count']} tests. \t Representative: {info['representative']}")

    with open(output_filtered_json, "w") as f:
        json.dump(clusters_filtered, f, indent=4)
    print(f"\nFiltered cluster info saved to {output_filtered_json}")

    # 5.2 打印并保存全量数据分配的结果
    print("\n=== Clustering Representatives (All Data) ===")
    for cid, info in clusters_all.items():
        rep_name = info['representative'] if info['representative'] else "None"
        print(f"Cluster {cid}: {info['count']} tests. \t Representative: {rep_name}")

    with open(output_all_json, "w") as f:
        json.dump(clusters_all, f, indent=4)
    print(f"\nAll data cluster info saved to {output_all_json}")

    # 6. 可视化（默认展示用过滤数据训练出来的 T-SNE 图）
    analyzer.visualize_tsne(output_file="tsne_cluster_filtered.png")