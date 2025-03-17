import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ----------------- 全局配置 -----------------
# 根据您提供的真实默认参数
DEFAULT_PARAM = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]
# --- 新增：全局字体与排版设置 ---
plt.rcParams.update({
    'font.size': 14,          # 全局默认字号变大
    'axes.titlesize': 16,     # 图表标题字号
    'axes.labelsize': 14,     # 坐标轴标签字号 (如 Normalized Speedup)
    'xtick.labelsize': 12,    # X轴刻度字号
    'ytick.labelsize': 12,    # Y轴刻度字号
    'legend.fontsize': 11,    # 图例字号（适度缩小，适配新布局）
    'font.family': 'serif',   # 强烈建议：改为衬线字体，与 LaTeX 论文风格更搭
    'font.serif': ['Times New Roman'],
    'axes.spines.top': False, # 隐藏顶部边框（可选，更简洁）
    'axes.spines.right': False# 隐藏右侧边框（可选，更简洁）
})

# ----------------- 数据加载 -----------------
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

# ----------------- 案例一：大偏离 + 大差异 + 高加速 -----------------
def find_extreme_adaptability_case(default_scores, final_data):
    """
    在 best 文件中寻找：加速比 > 20x，且修改了 >= 4个参数的测试，选出参数差异最大的一对。
    """
    candidates = []
    for d in final_data:
        name = d['litmus']
        final_score = d['score']
        def_score = default_scores.get(name, 0.0)

        # 兜底计算，防止除以 0
        safe_def_score = max(def_score, 0.01)
        speedup = final_score / safe_def_score

        # 筛选：加速比高
        if speedup > 20.0:
            # 筛选：修改的参数个数多
            mutated_count = sum([1 for p, d_p in zip(d['param'], DEFAULT_PARAM) if p != d_p])
            if mutated_count >= 4:
                candidates.append({
                    'litmus': name,
                    'param': d['param'],
                    'speedup': speedup,
                    'score': final_score
                })

    # 寻找参数配置距离最大的一对
    max_dist = 0
    best_pair = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            p1 = np.array(candidates[i]['param'])
            p2 = np.array(candidates[j]['param'])
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                max_dist = dist
                best_pair = (candidates[i], candidates[j])

    if best_pair:
        print(f"\n【案例一：极限偏离雷达图】挖掘成功！")
        print(f"Test A: {best_pair[0]['litmus']}")
        print(
            f" -> 原始分: {best_pair[0]['score']:.2f}, 加速比: {best_pair[0]['speedup']:.2f}x, params: {best_pair[0]['param']}")
        print(f"Test B: {best_pair[1]['litmus']}")
        print(
            f" -> 原始分: {best_pair[1]['score']:.2f}, 加速比: {best_pair[1]['speedup']:.2f}x, params: {best_pair[1]['param']}")
    else:
        print("\n未找到满足条件(加速比>20x 且 修改>=4个参数)的组合，可适当调低阈值。")

    return best_pair

# ----------------- 案例三：严格消融参数联动效应 (Synergy 1+1>2) -----------------
def find_synergy_case(default_scores, history_data, final_data):
    """
    寻找联动效应 (Synergy 1+1>2)：
    基于消融实验 (Ablation Study) 逻辑：
    最优配置 best_p 的加速比极高(>20x)，但去 history 里翻找时发现：
    如果保持其他参数都是最佳值，仅仅把参数 A 退回默认值，性能极差 (< 5x)；
    如果保持其他参数都是最佳值，仅仅把参数 B 退回默认值，性能同样极差 (< 5x)。
    """
    # 1. 整理 history 数据，方便精准查找完整的 param array
    test_history = {}
    for d in history_data:
        name = d['litmus']
        if name not in test_history:
            test_history[name] = []
        # 直接把 param 转成 tuple 作为 key，查询速度 O(1)
        test_history[name].append({
            'param': tuple(d['param']),
            'speedup': d['score']
        })

    # 2. 遍历 best 结果寻找符合条件的目标
    for d in history_data:
        name = d['litmus']
        final_score = d['score']
        def_score = default_scores.get(name, 0.0)

        safe_def_score = max(def_score, 0.01)
        best_speedup = final_score / safe_def_score

        # 前提：联合优化的最终加速比必须非常惊艳
        if best_speedup < 20.0:
            continue

        best_p = d['param']
        mutated_indices = [idx for idx, val in enumerate(best_p) if val != DEFAULT_PARAM[idx]]

        # 必须修改了至少 2 个参数才能谈联动
        if len(mutated_indices) < 2:
            continue

        records = test_history.get(name, [])

        # 两两组合寻找联动证据
        for i in range(len(mutated_indices)):
            for j in range(i + 1, len(mutated_indices)):
                idx_A = mutated_indices[i]
                idx_B = mutated_indices[j]

                # 构造 Ablation A：把最佳配置中的 A 退回默认值
                param_ablated_A = list(best_p)
                param_ablated_A[idx_A] = DEFAULT_PARAM[idx_A]
                param_ablated_A_tuple = tuple(param_ablated_A)

                # 构造 Ablation B：把最佳配置中的 B 退回默认值
                param_ablated_B = list(best_p)
                param_ablated_B[idx_B] = DEFAULT_PARAM[idx_B]
                param_ablated_B_tuple = tuple(param_ablated_B)

                # 在历史中精准匹配这两个 Ablation 配置
                speedup_ablated_A = None
                speedup_ablated_B = None

                for r in records:
                    if r['param'] == param_ablated_A_tuple:
                        speedup_ablated_A = r['speedup']
                    if r['param'] == param_ablated_B_tuple:
                        speedup_ablated_B = r['speedup']

                # 验证 Synergy (消融后性能出现断崖式下跌)
                if speedup_ablated_A is not None and speedup_ablated_B is not None:
                    # 如果缺了A或者缺了B，加速比都被打回原形 (< 5x)
                    if speedup_ablated_A < 10.0 and speedup_ablated_B < 10.0:
                        if final_score < 3 * (1 + speedup_ablated_A) * (speedup_ablated_B):
                            continue
                        print(f"\n【案例三：非线性参数联动 Synergy】严格消融挖掘成功！")
                        print(f"Test: {name}")
                        print(f"核心联动参数维度: 索引 [{idx_A}] 和 [{idx_B}]")
                        print(f" - Default Baseline: 1.00x")
                        print(f" - 最优配置 (Best): {best_p} -> 爆砍 {final_score:.2f}x 加速比！")
                        print(
                            f" - 剥离测试 A: 退回参数[{idx_A}]至默认值 ({param_ablated_A}) -> 暴跌至 {speedup_ablated_A:.2f}x")
                        print(
                            f" - 剥离测试 B: 退回参数[{idx_B}]至默认值 ({param_ablated_B}) -> 暴跌至 {speedup_ablated_B:.2f}x")

                        return name, idx_A, idx_B, speedup_ablated_A, speedup_ablated_B, final_score

    print("\n未找到完美的严格消融 Synergy 证据。可以尝试放宽阈值（比如允许退回默认时的加速比<10.0）")
    return None

# ----------------- 绘图功能 -----------------
def plot_synergy_bar(name, pA_name, pB_name, scoreA, scoreB, scoreAB):
    # 优化1：调整尺寸为 (6, 4.5)，与雷达图高度一致（4.5），宽度适配
    plt.figure(figsize=(6, 4.5))
    configs = ['Default\nBaseline', f'Only Change \n{pB_name}', f'Only Change \n{pA_name}',
               f'All Change\n']
    scores = [1.0, scoreA, scoreB, scoreAB]

    # 修改1：和左图统一配色，较差用蓝(#1f77b4)，最好用橙(#ff7f0e)
    colors = ['gray', '#1f77b4', '#C6DBEC', '#F9DFC7']

    bars = plt.bar(configs, scores, color=colors, width=0.6)

    # 修改2：去掉 log 坐标系，改回线性（绝对值）
    # plt.yscale('log')
    plt.ylabel('Normalized Speedup', fontsize=12)

    # 修改3：去掉标题中的 "Ablation Study"
    plt.title(name, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 修改4：设置动态 Y 轴上限，防止最优配置上方的数字被遮挡
    max_score = max(scores)
    plt.ylim(0, max_score * 1.15)

    for bar in bars:
        yval = bar.get_height()
        # 调整文本的高度偏移，适应线性坐标系
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max_score * 0.02),
                 f'{yval:.1f}x', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("case2_synergy_bar.pdf", format='pdf', bbox_inches='tight')
    plt.close()

def plot_radar_chart(test_A, test_B, max_params):
    test_A['param'] = test_A['param'][:-1]
    test_B['param'] = test_B['param'][:-1]
    categories = [f"P{i}" for i in range(len(test_A['param']))]
    N = len(categories)
    print(N)
    values_A = [p / m if m > 0 else 0 for p, m in zip(test_A['param'], max_params)]
    values_B = [p / m if m > 0 else 0 for p, m in zip(test_B['param'], max_params)]

    values_A += values_A[:1]
    values_B += values_B[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # 优化2：调整雷达图尺寸为 (5, 4.5)，高度和柱状图一致（4.5），宽度稍窄
    fig, ax = plt.subplots(figsize=(5, 4.5), subplot_kw=dict(polar=True))

    plt.xticks(angles[:-1], categories, fontsize=11)
    # 优化3：添加径向刻度，增强可读性
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'],
               fontsize=10, alpha=0.7)
    ax.set_rlabel_position(30) # 径向标签位置调整

    # 绘制两条曲线
    line1 = ax.plot(angles, values_A, linewidth=2, linestyle='solid',
                    color='#1f77b4', label=test_A['litmus'][:15] + '...')
    ax.fill(angles, values_A, alpha=0.25, color='#1f77b4')
    line2 = ax.plot(angles, values_B, linewidth=2, linestyle='solid',
                    color='#ff7f0e', label=test_B['litmus'][:15] + '...')
    ax.fill(angles, values_B, alpha=0.25, color='#ff7f0e')

    # 优化4：图例移到右侧，垂直排列，不遮挡图表
    plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.05), ncol=1,
               frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig("case1_radar.pdf", format='pdf', bbox_inches='tight')
    plt.close()

# ----------------- 主程序入口 -----------------
if __name__ == "__main__":
    # 请确保这三个文件路径正确
    default_file = '../RQ1/baseline_scores.json'
    history_file = '../RQ1/cache_norm_final.jsonl'
    final_best_file = '../RQ1/log_record_best.log.validation_cache.jsonl'

    # 加载数据
    default_scores = load_json(default_file)
    history_data = load_jsonl(history_file)
    final_data = load_jsonl(final_best_file)
    # 提取全局 max_params 用于雷达图归一化
    all_params = [d['param'] for d in final_data]
    max_params = np.max(all_params, axis=0) if all_params else [1] * 11
    max_params = [m if m > 0 else 1 for m in max_params]

    print(f"全局最大参数值 (用于归一化): {max_params}")

    # ================= 执行案例一：极限偏离雷达图 =================
    print("\n" + "=" * 50)
    print("开始挖掘案例一：参数极限偏离 (证明多样性)")
    pair = find_extreme_adaptability_case(default_scores, final_data)
    if pair:
        plot_radar_chart(pair[0], pair[1], max_params)
        print(">> 案例一雷达图已保存为 case1_radar.pdf")

    # ================= 执行案例三：参数联动 Synergy 柱状图 =================
    print("\n" + "=" * 50)
    print("开始挖掘案例三：严格消融参数联动 (证明 1+1>2 协同效应)")
    synergy_result = find_synergy_case(default_scores, history_data, final_data)

    if synergy_result:
        name, idx_A, idx_B, sA, sB, sAB = synergy_result

        # 将索引映射到真实的物理参数名，让生成的图表直接可用于论文
        PARAM_NAMES_MAP = [
            "mem",  # 0
            "barrier",  # 1
            "alloc",  # 2
            "detached",  # 3
            "thread",  # 4
            "launch",  # 5
            "affinity",  # 6
            "stride",  # 7
            "contiguous",  # 8
            "noalign",  # 9
            "param_10"  # 10
        ]

        # 安全获取参数名，如果超出范围就 fallback 到 P{idx}
        pA_name = PARAM_NAMES_MAP[idx_A] if idx_A < len(PARAM_NAMES_MAP) else f'P{idx_A}'
        pB_name = PARAM_NAMES_MAP[idx_B] if idx_B < len(PARAM_NAMES_MAP) else f'P{idx_B}'
        print(name, pA_name, pB_name, sA, sB, sAB)
        # 绘图
        plot_synergy_bar(name, pA_name, pB_name, sA, sB, sAB)
        print(">> 案例三协同柱状图已保存为 case2_synergy_bar.pdf")
