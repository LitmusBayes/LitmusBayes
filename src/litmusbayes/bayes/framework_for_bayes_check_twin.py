import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ================= 配置路径 =================
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache4_norm.jsonl"

# 输出文件 1: 去重后的列表 (只存每组第一个) - 用于后续训练
unique_save_path = "unique_vector_representatives.json"
# 输出文件 2: 冲突分组详情 (存每组的所有成员) - 用于分析哪些是一样的
collision_save_path = "identical_vector_groups.json"


def load_data():
    print("Loading Data...")

    # 1. 加载向量
    vectors = {}  # {name: np_array}
    with open(litmus_vec_path, "r") as f:
        for line in f:
            if ":" in line:
                name, vec_str = line.strip().split(":", 1)
                vectors[name] = np.array(eval(vec_str))

    # 2. 加载分数 {name: {param_tuple: score}}
    scores = defaultdict(dict)
    with open(cache_file_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                name = obj["litmus"]
                p = tuple(obj["param"])
                s = obj["score"]
                scores[name][p] = s
            except:
                pass

    # 取交集
    common_names = list(set(vectors.keys()) & set(scores.keys()))
    print(f"Loaded {len(common_names)} valid litmus tests.")
    return common_names, vectors, scores


def analyze_identical_vector_gap():
    names, vectors, scores = load_data()

    # === 第一步：寻找向量双胞胎 (Vector Twins) ===
    print("Grouping tests by Vector Identity...")

    # 为了加速，我们把 Vector 转成 tuple 作为 key
    vec_groups = defaultdict(list)

    for n in names:
        # 保留 4 位小数，视为“相同”
        v_key = tuple(np.round(vectors[n], 4))
        vec_groups[v_key].append(n)

    print(f"Found {len(vec_groups)} unique vector patterns.")

    # ================= 【保存逻辑】 =================

    # 1. 保存【去重后的代表列表】 (每组只存第一个)
    unique_representatives = []
    for group_list in vec_groups.values():
        if group_list:
            unique_representatives.append(group_list[0])  # 拿第一个

    with open(unique_save_path, "w") as f:
        json.dump(unique_representatives, f, indent=4)
    print(f"Saved {len(unique_representatives)} unique representatives to {unique_save_path}")

    # 2. 保存【完全相同的冲突组】 (包含每组的第一个和后续重复的)
    # 只有当一组里有 >1 个测试时，才算冲突
    collision_groups_list = [group for group in vec_groups.values() if len(group) > 1]

    with open(collision_save_path, "w") as f:
        json.dump(collision_groups_list, f, indent=4)
    print(f"Saved {len(collision_groups_list)} collision groups to {collision_save_path}")
    # ===============================================

    print(f"Found {len(collision_groups_list)} groups where multiple tests share the SAME vector.")

    # === 第二步：计算同参数下的分数差异 (保持不变) ===

    all_relative_diffs = []
    severe_cases = []

    print("Calculating performance gaps within twins...")

    for group in collision_groups_list:
        # 组内两两比较
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                name_a = group[i]
                name_b = group[j]

                params_a = set(scores[name_a].keys())
                params_b = set(scores[name_b].keys())
                common_params = params_a.intersection(params_b)

                if not common_params: continue

                for p in common_params:
                    score_a = scores[name_a][p]
                    score_b = scores[name_b][p]

                    # 噪声过滤
                    abs_diff = abs(score_a - score_b)
                    if abs_diff < 1e-3: continue

                    denom = max(abs(score_a), abs(score_b))
                    if denom < 1e-6: continue

                    rel_diff = abs_diff / denom
                    all_relative_diffs.append(rel_diff)

                    if rel_diff > 0.5:
                        severe_cases.append({
                            "Test A": name_a,
                            "Test B": name_b,
                            "Param": p,
                            "Score A": score_a,
                            "Score B": score_b,
                            "Rel Diff": rel_diff
                        })

    # === 第三步：结果分析 ===
    print("\n" + "=" * 60)
    print("      IDENTICAL VECTOR ANALYSIS REPORT      ")
    print("=" * 60)

    arr = np.array(all_relative_diffs)
    if len(arr) == 0:
        print("No common parameters found between twins.")
        return

    print(f"Total Comparisons: {len(arr)}")
    print(f"Mean Relative Difference:   {np.mean(arr) * 100:.2f}%")

    # 打印前几个严重案例
    if len(severe_cases) > 0:
        print("\n[SMOKING GUN] Top-3 Examples where Vector is SAME but Score is DIFFERENT:")
        severe_cases.sort(key=lambda x: x["Rel Diff"], reverse=True)
        for idx, case in enumerate(severe_cases[:3]):
            print(f"Case #{idx + 1}:")
            print(f"  Tests: {case['Test A']} vs {case['Test B']}")
            print(f"  Score: {case['Score A']:.4e} vs {case['Score B']:.4e}")
            print(f"  Diff:  {case['Rel Diff'] * 100:.1f}%")
            print("-" * 30)

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.histplot(arr, bins=50, kde=True, color='purple')
    plt.title("Performance Gap Distribution for Identical Vectors")
    plt.savefig("./analysis_plots/vector_collision_gap.png")
    print("Plot saved.")


if __name__ == "__main__":
    if not os.path.exists("./analysis_plots"): os.makedirs("./analysis_plots")
    analyze_identical_vector_gap()