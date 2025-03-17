import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import ast


# 1. 加载 Embeddings
def load_embeddings(filepath):
    names = []
    vectors = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # 格式: Name:[v1, v2, ...]
            if ":" in line:
                name, vec_str = line.split(":", 1)
                names.append(name)
                # 安全地将字符串列表转为 list
                vectors.append(ast.literal_eval(vec_str))

    return names, np.array(vectors)


# 2. 核心：计算相似度
def analyze_similarity(names, vectors):
    # 计算余弦相似度矩阵 (900 x 900)
    # 结果 sim_matrix[i][j] 表示第 i 个和第 j 个 Test 的相似度，范围 [-1, 1]
    sim_matrix = cosine_similarity(vectors)

    return sim_matrix


# 3. 查找 Top-K 相似
def print_top_k_similar(target_name, names, sim_matrix, k=5):
    try:
        target_idx = names.index(target_name)
    except ValueError:
        print(f"Error: {target_name} not found.")
        return

    # 获取该行相似度
    scores = sim_matrix[target_idx]

    # 排序 (从大到小)，排除自己 (自己和自己相似度是1)
    # argsort 返回的是索引，[::-1] 反转成从大到小
    sorted_indices = np.argsort(scores)[::-1]

    print(f"\n=== Top {k} neighbors for: {target_name} ===")
    count = 0
    for idx in sorted_indices:
        if idx == target_idx: continue  # 跳过自己

        neighbor_name = names[idx]
        score = scores[idx]
        print(f"{score:.4f} | {neighbor_name}")

        count += 1
        if count >= k: break


# 4. 可视化：t-SNE 降维
def plot_tsne(names, vectors, output_file="tsne_litmus.png"):
    print("\nRunning t-SNE (this might take a few seconds)...")
    # perplexity 建议设为 30-50，如果数据点很少(如<100)就设小一点
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    vec_2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(12, 10))
    plt.scatter(vec_2d[:, 0], vec_2d[:, 1], alpha=0.6, s=15, c='blue')

    # 可选：给部分点标上名字 (为了不拥挤，随机标几个或者不标)
    # for i, name in enumerate(names):
    #     if i % 20 == 0: # 每20个标一个名字
    #         plt.annotate(name, (vec_2d[i, 0], vec_2d[i, 1]), fontsize=8, alpha=0.7)

    plt.title("t-SNE Visualization of Litmus Test Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"t-SNE plot saved to {output_file}")
    plt.show()


# 5. 可视化：热力图 (展示局部相似性)
def plot_heatmap(sim_matrix, output_file="heatmap.png"):
    plt.figure(figsize=(10, 8))
    # 只画前 50 个，否则 900x900 太密了看不清
    subset_matrix = sim_matrix[:50, :50]
    sns.heatmap(subset_matrix, cmap="viridis")
    plt.title("Similarity Heatmap (First 50 Tests)")
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")


# ================= 运行脚本 =================

if __name__ == "__main__":
    embedding_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/learned_embeddings.txt"

    # A. 加载
    names, vectors = load_embeddings(embedding_path)
    print(f"Loaded {len(names)} vectors with dimension {vectors.shape[1]}")

    # B. 计算矩阵
    sim_matrix = analyze_similarity(names, vectors)

    # C. 随便挑一个测试查询相似度 (假设列表里有这个名字，你可以换成你真实的Litmus名)
    if len(names) > 0:
        target = names[0]
        print_top_k_similar(target, names, sim_matrix, k=5)

    # D. 画图
    # plot_heatmap(sim_matrix) # 可选
    plot_tsne(names, vectors)