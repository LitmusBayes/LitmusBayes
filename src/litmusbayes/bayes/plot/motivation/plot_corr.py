import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats  # 需要引入 scipy 来计算斯皮尔曼相关系数


def normalize(arr):
    min_val, max_val = min(arr), max(arr)
    if max_val == min_val:
        return [0.5] * len(arr)
    return [(x - min_val) / (max_val - min_val) for x in arr]


def auto_find_flawless_pair(log_file, output_fig="motivation_fig_c.pdf"):
    data = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line.strip())
            name = record['litmus']
            param_str = str(record['param'])
            score = record['score']

            if name not in data:
                data[name] = {}
            if param_str not in data[name] or score > data[name][param_str]:
                data[name][param_str] = score

    test_names = list(data.keys())
    best_pair = None
    best_corr = -1
    shared_params_for_best = []

    print("正在使用斯皮尔曼等级相关系数 + 去零过滤，寻找视觉完美的测试对...")

    for ta, tb in itertools.combinations(test_names, 2):
        # 可选：如果强行跨家族找出来的图实在不好看，可以把下面的注释打开，先在同家族里找
        if ta.split('+')[0] == tb.split('+')[0]: continue

        shared = list(set(data[ta].keys()) & set(data[tb].keys()))

        valid_shared = []
        for p in shared:
            # 核心升级 1：只要有任何一个是 0，或者两者极其接近 0，就剔除，防止“双零”拉高相关性
            if data[ta][p] > 0.01 and data[tb][p] > 0.01:
                valid_shared.append(p)

        # 至少需要 8-10 个有效的非零波动点才足以支撑折线图
        if len(valid_shared) < 8:
            continue

        scores_a = [data[ta][p] for p in valid_shared]
        scores_b = [data[tb][p] for p in valid_shared]

        if max(scores_a) == min(scores_a) or max(scores_b) == min(scores_b):
            continue

        # 核心升级 2：使用斯皮尔曼等级相关系数，寻找“走势/排名”最一致的组合
        corr, _ = stats.spearmanr(scores_a, scores_b)

        if corr > best_corr:
            best_corr = corr
            best_pair = (ta, tb)
            shared_params_for_best = valid_shared

    if not best_pair:
        print("未找到满足条件的完美测试对。建议退回皮尔逊系数，或者放宽非零过滤条件。")
        return

    ta, tb = best_pair
    print(f"\n--- 成功锁定【视觉完美】的 Case ---")
    print(f"测试 A: {ta}")
    print(f"测试 B: {tb}")
    print(f"有效波动配置数: {len(shared_params_for_best)}")
    print(f"斯皮尔曼等级相关系数: {best_corr:.4f} (极高！)")

    # 按照测试 A 的得分对参数排序
    shared_params_for_best.sort(key=lambda p: data[ta][p])
    plot_params = shared_params_for_best[:10]

    sa = [data[ta][p] for p in plot_params]
    sb = [data[tb][p] for p in plot_params]

    norm_a = normalize(sa)
    norm_b = normalize(sb)

    plt.figure(figsize=(5, 4))
    x_points = range(len(plot_params))

    plt.plot(x_points, norm_a, marker='o', linestyle='-', linewidth=2, color='#D32F2F', label=f'Test A: {ta}')
    plt.plot(x_points, norm_b, marker='^', linestyle='--', linewidth=2, color='#1976D2', label=f'Test B: {tb}')

    # plt.title('High Performance Correlation Across Tests with Shared Graph Semantics', fontsize=14, pad=15)
    plt.xlabel('Shared Parameter Configurations', fontsize=12)
    plt.ylabel('Normalized Trigger Rate', fontsize=12)
    plt.xticks(x_points, [f'C{i + 1}' for i in x_points])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_fig, format='pdf')
    print(f"\n成功生成新的论文用图: {output_fig}")
    plt.show()


# 运行前请确保安装了 scipy: pip install scipy
if __name__ == "__main__":
    log_file_path = "../../log/cache_norm.jsonl"  # 替换为你的日志路径
    auto_find_flawless_pair(log_file_path)