import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

# ==========================================
# 0. 设置学术论文绘图风格
# ==========================================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.size': 11,         # 基础字号
    'axes.labelsize': 12,    # 坐标轴标签字号
    # 'axes.titlesize': 12,  # 标题反正不画了，注释掉
    'xtick.labelsize': 10,   # X轴刻度字号
    'ytick.labelsize': 10,   # Y轴刻度字号
    'legend.fontsize': 10,   # 图例字号
    'pdf.fonttype': 42,      # 保证导出矢量图里的字体兼容性
    'ps.fonttype': 42
})
# ==========================================
# 1. 加载和预处理数据
# ==========================================
print("Loading data...")

# 1.1 加载基线分数 (假设 baseline_scores.json 是 {"test_name": score} 的格式)
with open('../../log/baseline_scores.json', 'r') as f:
    baseline_scores = json.load(f)

# 按照默认难度降序排序
sorted_tests = sorted(baseline_scores.keys(), key=lambda x: baseline_scores[x], reverse=True)
base_scores_list = [baseline_scores[t] for t in sorted_tests]

# 1.2 加载全部采样的结果 (cache.jsonl)
best_scores_dict = {t: baseline_scores.get(t, 0) for t in sorted_tests}

# 选定一个用来画图 (c) 非线性特征的代表性困难测试
# 请替换为您数据里真实存在的困难测试名字
target_test_for_c = "SB+pos-po+pos-po-ctrlfenceis"
if target_test_for_c not in sorted_tests:
    target_test_for_c = sorted_tests[-1]  # 如果没有，就选最难的那一个

stride_data_for_c = []  # 存 [(特定参数值, 得分), ...]

with open('../../log/cache_norm.jsonl', 'r') as f:
    for line in f:
        if not line.strip(): continue
        try:
            data = json.loads(line.strip())
            t_name = data['litmus']
            score = data['score']
            params = data['param']

            # 更新每个测试的历史最高分
            if t_name in best_scores_dict and score * baseline_scores[t_name] > best_scores_dict[t_name]:
                best_scores_dict[t_name] = score * baseline_scores[t_name]

            # 收集特定测试的特定参数影响 (例如假设 params[6] 是 stride 或重要参数)
            if t_name == target_test_for_c:
                target_param_val = params[6] if len(params) > 6 else len(stride_data_for_c)
                stride_data_for_c.append((target_param_val, score))

        except (json.JSONDecodeError, KeyError) as e:
            continue

best_scores_list = [best_scores_dict[t] for t in sorted_tests]

# 1.3 加载归一化后的数据 (cache_norm.jsonl)
norm_data_records = []
with open('../../log/cache_norm.jsonl', 'r') as f:  # 注意这里我改成了 .jsonl
    for line in f:
        if not line.strip(): continue
        try:
            data = json.loads(line.strip())
            t_name = data['litmus']
            norm_score = data['score']  # 在这个文件里 score 代表加速比

            norm_data_records.append({"Test": t_name, "Normalized Speedup": norm_score, "params": data['param']})
        except (json.JSONDecodeError, KeyError):
            continue

df_norm = pd.DataFrame(norm_data_records)

# ==========================================
# 2. 开始绘制并分开保存子图
# ==========================================
print("Plotting and saving individual figures...")

# 单个图的尺寸，你可以根据论文单栏或双栏宽度自己微调
single_figsize = (5.0, 4.0)

# ---------------------------------------------------------
# Figure (a): 绝对难度 vs. 调优潜力 (散点图)
# ---------------------------------------------------------
fig_a, ax_a = plt.subplots(figsize=single_figsize)

x_pos = np.arange(len(sorted_tests))
# 画默认参数灰点 & 最优参数红星
ax_a.scatter(x_pos, base_scores_list, color='#7f7f7f', label='Default Params (Baseline)', s=15, alpha=0.6, zorder=3)
ax_a.scatter(x_pos, best_scores_list, color='#d62728', marker='*', label='Best Found Params', s=30, zorder=3)

# 画垂直连线
lines = [[(i, base_scores_list[i]), (i, best_scores_list[i])] for i in range(len(sorted_tests))]
lc = LineCollection(lines, colors='black', linestyles=':', alpha=0.15)
ax_a.add_collection(lc)

ax_a.set_yscale('log')
ax_a.set_xlim(-10, len(sorted_tests) + 10)
ax_a.set_xticks([])  # 隐藏密集的X轴标签
ax_a.set_xlabel(f'{len(sorted_tests)} Litmus Tests (Sorted by Inherent Difficulty)')
ax_a.set_ylabel('Triggers / Second (Log Scale)')
# ax_a.set_title('(a) Inherent Difficulty vs. Tuning Potential')
ax_a.legend(loc='upper right')

plt.tight_layout()
fig_a.savefig('motivation_fig_a.pdf', format='pdf', bbox_inches='tight')
print("Saved motivation_fig_a.pdf")
plt.close(fig_a)


# ---------------------------------------------------------
# Figure (b): 归一化加速比箱线图
# ---------------------------------------------------------
fig_b, ax_b = plt.subplots(figsize=single_figsize)

# # 1. 定义需要寻找的前缀目标
selected_tests = []
selected_test_names = ['ISA12', 'R+rfi-ctrl+rfi-addr', 'MP+fence.rw.w+po', 'SB+fence.rw.rw+rfi-ctrl-rfi', 'S+rfi-data+fence.r.rw', 'LB+data+po']
# 2. 遍历 sorted_tests，为每个前缀寻找第一个匹配的 test
for name in selected_test_names:
    for test in sorted_tests:
        # 使用 split('+')[0] 进行精准匹配，避免例如 'S' 错误匹配到 'SB'
        if test == name:
            selected_tests.append(test)
            break  # 找到一个后立刻跳出内层循环，寻找下一个前缀

print("Selected tests:", selected_tests)




if not df_norm.empty:
    df_norm_filtered = df_norm[df_norm['Test'].isin(selected_tests)].copy()
    print(df_norm_filtered)
    # 截断超长测试名字
    df_norm_filtered['Short_Test'] = df_norm_filtered['Test'].apply(lambda x: x.split('+')[0] + "+...")

    sns.boxplot(data=df_norm_filtered, x="Short_Test", y="Normalized Speedup",
                ax=ax_b, color="#aec7e8", fliersize=3, linewidth=1.2)

ax_b.axhline(1.0, color='#d62728', linestyle='--', label='Default Baseline (1.0x)')
ax_b.set_yscale('log')
ax_b.set_ylim(1e-2, 1e2)
ax_b.tick_params(axis='x', rotation=30)
ax_b.set_xlabel('')
# ax_b.set_title('(b) Speedup Distribution')
ax_b.legend(loc='upper right')

plt.tight_layout()
fig_b.savefig('motivation_fig_b.pdf', format='pdf', bbox_inches='tight')
print("Saved motivation_fig_b.pdf")
plt.close(fig_b)


# ---------------------------------------------------------
# Figure (c): 参数的非线性敏感度
# ---------------------------------------------------------
# fig_c, ax_c = plt.subplots(figsize=single_figsize)
#
# if stride_data_for_c:
#     stride_data_for_c.sort(key=lambda x: x[0])
#     df_c = pd.DataFrame(stride_data_for_c, columns=['Param_Val', 'Score'])
#     df_c_mean = df_c.groupby('Param_Val').mean().reset_index()
#
#     ax_c.plot(df_c_mean['Param_Val'], df_c_mean['Score'], marker='o', linestyle='-', color='#9467bd', linewidth=1.5,
#                  markersize=4)
#
#     base_val = baseline_scores.get(target_test_for_c, 0)
#     if base_val > 0:
#         ax_c.axhline(base_val, color='gray', linestyle='--', label=f'Default ({base_val:.1f})')
#
#     ax_c.set_xlabel('Parameter Dimension (e.g., param[6])')
#     ax_c.set_ylabel('Triggers / Second')
#     short_c_name = target_test_for_c.split('+')[0] + "+..."
#     # ax_c.set_title(f'(c) Non-linear Sensitivity ({short_c_name})')
#     ax_c.legend(loc='upper right')
# else:
#     ax_c.text(0.5, 0.5, "No Data for Plot (c)", ha='center', va='center')

# plt.tight_layout()
# fig_c.savefig('motivation_fig_c.jpg', format='jpg', bbox_inches='tight', dpi=300)
# print("Saved motivation_fig_c.jpg")
# plt.close(fig_c)

print("All figures separated and saved successfully!")