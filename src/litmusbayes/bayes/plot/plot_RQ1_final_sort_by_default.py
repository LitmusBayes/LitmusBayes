import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置文件路径 (请确保路径与你本地一致)
# ==========================================

BASELINE_FILE = '../experiment/RQ1/final_result/init/median_results.json'
PERPLE_FILE = "../experiment/perple/C910_log/median_results.json"
BAYES_FILE = '../experiment/RQ1/final_result/bayes_final/median_results.json'

# ==========================================
# 2. 加载数据
# ==========================================
baseline_data = {}
with open(BASELINE_FILE, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            baseline_data[item['litmus']] = item['score']

with open(PERPLE_FILE, 'r') as f:
    perple_data = json.load(f)

bayes_data = {}
with open(BAYES_FILE, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            bayes_data[item['litmus']] = item['score']

# ==========================================
# 3. 数据对齐与计算 (按 base_score 从小到大排序)
# ==========================================
# 获取所有测试用例的集合
all_tests = list(set(baseline_data.keys()) | set(bayes_data.keys()) | set(perple_data.keys()))

# 【核心修改点】：按照 base_score (baseline_data 中的分数) 从小到大排序
# 如果某个 test 不在 baseline_data 中，默认分数为 0，会排在最前面
all_tests.sort(key=lambda test: baseline_data.get(test, 0))

MIN_SPEEDUP = 1e-4

speedup_bayes_p1, speedup_perple_p1 = [], []
freq_baseline_p2, freq_bayes_p2, freq_perple_p2 = [], [], []

for test in all_tests:
    base_score = baseline_data.get(test, 0)
    bayes_score = bayes_data.get(test, 0)
    perple_score = perple_data.get(test, 0)

    # 图 2 逻辑
    if base_score > 0 or bayes_score > 0 or perple_score > 0:
        freq_baseline_p2.append(base_score if base_score > 0 else MIN_SPEEDUP)
        freq_bayes_p2.append(bayes_score if bayes_score > 0 else MIN_SPEEDUP)
        freq_perple_p2.append(perple_score if perple_score > 0 else MIN_SPEEDUP)

    # 图 1 逻辑
    if base_score > 0:
        sb = bayes_score / base_score
        sp = perple_score / base_score
        speedup_bayes_p1.append(max(sb, MIN_SPEEDUP) if bayes_score > 0 else MIN_SPEEDUP)
        speedup_perple_p1.append(max(sp, MIN_SPEEDUP) if perple_score > 0 else MIN_SPEEDUP)

speedup_bayes_p1 = np.array(speedup_bayes_p1)
speedup_perple_p1 = np.array(speedup_perple_p1)

freq_baseline_p2 = np.array(freq_baseline_p2)
freq_bayes_p2 = np.array(freq_bayes_p2)
freq_perple_p2 = np.array(freq_perple_p2)

num_tests_p1 = len(speedup_bayes_p1)
num_tests_p2 = len(freq_baseline_p2)

if num_tests_p2 == 0:
    print("Error: No data left to plot! Exiting...")
    exit()

bayes_accel_count = np.sum(speedup_bayes_p1 > 1)
perple_accel_count = np.sum(speedup_perple_p1 > 1)

# ==========================================
# 4. 绘图 (已排序面积图 + 边缘描边技术)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')

# 配色
c_bayes = '#1f77b4'   # 深蓝色 (LitmusBayes)
c_perple = '#ff7f0e'  # 亮橙色 (perple)
c_base = '#2ca02c'    # 翠绿色 (Baseline)


# ------------------------------------------
# 图 1：加速比面积图 (Sorted Speedup Area)
# ------------------------------------------
if num_tests_p1 > 0:
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    x_p1 = np.arange(1, num_tests_p1 + 1)

    PLOT_MIN_1 = 0.1
    plot_bayes_1 = np.clip(speedup_bayes_p1, PLOT_MIN_1, None)
    plot_perple_1 = np.clip(speedup_perple_p1, PLOT_MIN_1, None)

    ax1.axhline(y=1, color='black', linestyle='--', linewidth=2, zorder=5, label='litmus7 (Baseline, 1x)')


    ax1.fill_between(x_p1, PLOT_MIN_1, plot_perple_1, color=c_perple, alpha=0.3, zorder=1)
    ax1.plot(x_p1, plot_perple_1, color=c_perple, alpha=0.9, linewidth=1.5, zorder=2,
             label=f"PerpLE (MICRO'20) [{perple_accel_count} accelerated]")

    ax1.fill_between(x_p1, PLOT_MIN_1, plot_bayes_1, color=c_bayes, alpha=0.4, zorder=3)
    ax1.plot(x_p1, plot_bayes_1, color=c_bayes, alpha=1.0, linewidth=1.5, zorder=4,
             label=f"LitmusBayes [{bayes_accel_count} accelerated]")


    ax1.set_yscale('log')
    # 【修改点】：更新 X 轴标签描述
    ax1.set_xlabel('Litmus Tests', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speedup over litmus7 (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, num_tests_p1)
    ax1.set_ylim(bottom=PLOT_MIN_1)

    ax1.text(1, PLOT_MIN_1 * 1.2, 'Failed / Not Triggered / <0.1x', color='black', alpha=0.7, fontsize=10)
    ax1.legend(fontsize=11, frameon=True, edgecolor='black', loc='upper left')

    fig1.tight_layout()
    # 建议此处如果你想区分文件的话可以改名，如改成 _sorted.png，此处保留原名但稍微改了后缀
    fig1.savefig('../experiment/results/litmus_evaluation_speedup_sorted.pdf', format='pdf')
    print("Saved: litmus_evaluation_speedup_sorted.pdf")

# ------------------------------------------
# 图 2：绝对触发频率面积图 (Sorted Absolute Triggers)
# ------------------------------------------
fig2, ax2 = plt.subplots(figsize=(12, 5))
x_p2 = np.arange(1, num_tests_p2 + 1)

PLOT_MIN_2 = 0.5
plot_base_2 = np.clip(freq_baseline_p2, PLOT_MIN_2, None)
plot_perple_2 = np.clip(freq_perple_p2, PLOT_MIN_2, None)
plot_bayes_2 = np.clip(freq_bayes_p2, PLOT_MIN_2, None)

ax2.fill_between(x_p2, PLOT_MIN_2, plot_base_2, color=c_base, alpha=0.3, zorder=1)
ax2.plot(x_p2, plot_base_2, color=c_base, alpha=0.9, linewidth=1.5, zorder=2, label="litmus7 (Baseline)")

ax2.fill_between(x_p2, PLOT_MIN_2, plot_perple_2, color=c_perple, alpha=0.3, zorder=3)
ax2.plot(x_p2, plot_perple_2, color=c_perple, alpha=0.9, linewidth=1.2, zorder=4, label="PerpLE (MICRO'20)")

ax2.fill_between(x_p2, PLOT_MIN_2, plot_bayes_2, color=c_bayes, alpha=0.4, zorder=5)
ax2.plot(x_p2, plot_bayes_2, color=c_bayes, alpha=1.0, linewidth=1.5, zorder=6, label="LitmusBayes")

ax2.set_yscale('log')
# 【修改点】：更新 X 轴标签描述
ax2.set_xlabel('Litmus Tests', fontsize=12, fontweight='bold')
ax2.set_ylabel('Trigger Rate (Log Scale)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, num_tests_p2)
ax2.set_ylim(bottom=PLOT_MIN_2)

ax2.axhline(y=1, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.text(1, PLOT_MIN_2 * 1.2, 'Failed / Not Triggered (Value < 1)', color='black', alpha=0.7, fontsize=10)
ax2.legend(fontsize=11, frameon=True, edgecolor='black', loc='upper left') # 调整了 legend 位置防止遮挡排序后的尾部高点

fig2.tight_layout()
fig2.savefig('../experiment/results/litmus_evaluation_triggers_sorted.pdf', format='pdf')
print("Saved: litmus_evaluation_triggers_sorted.pdf")

plt.show()