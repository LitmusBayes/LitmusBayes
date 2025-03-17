import matplotlib.pyplot as plt
import numpy as np

# 1. 设置干净的绘图环境
fig, ax = plt.subplots(figsize=(2, 2))

# 2. 生成合成的 k-means 聚类数据
np.random.seed(123)

# 定义簇中心 (x, y)
centroid_1_pos = np.array([3, 7])
centroid_2_pos = np.array([7, 7])
centroid_3_pos = np.array([3, 3])

# 在中心周围生成随机点
n_points_per_cluster = 15
std_dev = 0.80

data_1 = np.random.normal(loc=centroid_1_pos, scale=std_dev, size=(n_points_per_cluster, 2))
data_2 = np.random.normal(loc=centroid_2_pos, scale=std_dev, size=(n_points_per_cluster, 2))
data_3 = np.random.normal(loc=centroid_3_pos, scale=std_dev, size=(n_points_per_cluster, 2))

centroids = np.array([centroid_1_pos, centroid_2_pos, centroid_3_pos])

# 3. 绘图设置
cluster_colors = ['#E57373', '#6BAB64', '#64B5F6']
centroid_color = '#212121'

# 设置指定的背景颜色
bg_color = '#CCE8E1'
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# 彻底删除所有边框 (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

# 删除所有刻度和标签
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# 设置一个固定的轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 4. 绘制数据点
all_data = [data_1, data_2, data_3]
for i, data in enumerate(all_data):
    ax.scatter(data[:, 0], data[:, 1], c=cluster_colors[i], alpha=0.7, edgecolors='none', s=60, zorder=1)

# 5. 绘制质心
ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', color=centroid_color, s=100, zorder=10)

# 6. 调整布局并显示
plt.tight_layout()
# plt.show()

# 如果需要保存图片，请取消下面这行的注释（已包含背景色设置）
plt.savefig('improved_k_means_plot.png', dpi=300, facecolor=bg_color, bbox_inches='tight', pad_inches=0.1)