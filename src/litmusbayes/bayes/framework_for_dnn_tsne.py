import numpy as np
import pandas as pd
import ast
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px


# 1. 加载 Embeddings (保持不变)
def load_embeddings(filepath):
    names = []
    vectors = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if ":" in line:
                name, vec_str = line.split(":", 1)
                names.append(name)
                vectors.append(ast.literal_eval(vec_str))
    return names, np.array(vectors)


# 2. 交互式 t-SNE 可视化 (核心修改)
def plot_interactive_tsne(names, vectors, output_file="litmus_map.html"):
    print("\nRunning t-SNE (calculating 2D coordinates)...")
    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    vec_2d = tsne.fit_transform(vectors)

    print("Running K-Means (grouping similar tests by color)...")
    # 自动聚类：这里假设聚成 10 类 (你可以改 n_clusters)，方便看颜色区分
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vectors)  # 注意：我们在原始高维向量上聚类，效果更准

    # 构建 DataFrame，方便 Plotly 绘图
    df = pd.DataFrame({
        'x': vec_2d[:, 0],
        'y': vec_2d[:, 1],
        'Litmus Name': names,
        'Cluster': [f"Group {c}" for c in clusters]  # 转成字符串以便作为离散颜色
    })

    print("Generating HTML plot...")
    # 使用 Plotly 画图
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Cluster',  # 根据聚类结果自动着色
        hover_name='Litmus Name',  # 【关键】鼠标悬停显示 Litmus 名字
        title='Interactive Map of Litmus Tests (t-SNE)',
        template='plotly_white',  # 使用简洁的白色背景
        width=1200, height=800
    )

    # 优化点的样式
    fig.update_traces(marker=dict(size=8, opacity=0.8))

    # 隐藏坐标轴刻度 (因为 t-SNE 的坐标数值本身没有物理意义)
    fig.update_xaxes(showticklabels=False, title='')
    fig.update_yaxes(showticklabels=False, title='')

    # 保存为 HTML
    fig.write_html(output_file)
    print(f"\nSuccess! Open this file in your browser: {output_file}")


# ================= 运行 =================

if __name__ == "__main__":
    embedding_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/learned_embeddings.txt"

    # 加载
    names, vectors = load_embeddings(embedding_path)
    print(f"Loaded {len(names)} vectors.")

    if len(names) > 0:
        # 运行交互式可视化
        plot_interactive_tsne(names, vectors, output_file="litmus_map.html")
    else:
        print("Error: No data found.")