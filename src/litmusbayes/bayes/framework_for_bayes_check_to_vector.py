import json
import numpy as np
import os
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================= 配置 =================
cache_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_norm_for_graph.jsonl"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"


def analyze_vector_quality():
    print("=== 开始 Vector 质量深度解剖 ===")

    # 1. 加载 Vector
    litmus_to_vec = {}
    vec_to_litmus_list = defaultdict(list)

    if os.path.exists(litmus_vec_path):
        with open(litmus_vec_path, "r") as f:
            for line in f:
                if ":" in line:
                    try:
                        n, v = line.strip().split(":", 1)
                        vec_tuple = tuple(eval(v))  # 转成 tuple 才能做 key
                        litmus_to_vec[n] = vec_tuple
                        vec_to_litmus_list[vec_tuple].append(n)
                    except:
                        pass

    total_files = len(litmus_to_vec)
    unique_vectors = len(vec_to_litmus_list)

    print(f"\n[1. 唯一性检查]")
    print(f"总文件数: {total_files}")
    print(f"唯一 Vector 数: {unique_vectors}")
    print(f"重复率: {1 - unique_vectors / total_files:.2%}")

    # 找出撞衫最严重的 Vector
    print("\n[2. 撞衫之王 (Top 5 拥挤的 Vector)]")
    sorted_vecs = sorted(vec_to_litmus_list.items(), key=lambda x: len(x[1]), reverse=True)
    for i in range(min(9, len(sorted_vecs))):
        vec, files = sorted_vecs[i]
        print(f"Rank {i + 1}: 有 {len(files)} 个文件共用此 Vector!")
        print(f"  -> Vector: {vec}")
        print(f"  -> 示例文件: {files[:3]}...")

    # 3. 维度方差检查
    print("\n[3. 维度有效性检查]")
    # 转成矩阵
    all_vecs = np.array(list(litmus_to_vec.values()))
    if all_vecs.shape[0] > 0:
        std_per_dim = np.std(all_vecs, axis=0)
        mean_per_dim = np.mean(all_vecs, axis=0)

        print(f"Vector 维度: {all_vecs.shape[1]}")
        print(f"{'Dim':<5} | {'Mean':<10} | {'Std (方差)':<10} | {'状态'}")
        print("-" * 45)
        for i in range(len(std_per_dim)):
            status = "✅ 正常"
            if std_per_dim[i] < 1e-6:
                status = "❌ 死特征 (全是常量)"
            elif std_per_dim[i] < 0.01:
                status = "⚠️ 极低方差 (几乎没变)"
            print(f"{i:<5} | {mean_per_dim[i]:<10.4f} | {std_per_dim[i]:<10.4f} | {status}")

    # 4. 加载 Score 数据，检查“同 Vector 不同命”的现象
    print("\n[4. 致命冲突检查 (Same Vector, Different Best Param)]")
    # 逻辑：找出共用同一个 Vector 的文件，看看它们的最优参数是不是不一样
    # 如果 Vector 一样，但一个喜欢参数A，一个喜欢参数B，那就是彻底的冲突

    # 加载每个文件的最优参数
    file_best_param = {}
    with open(cache_file_path, "r") as f:
        # 先把所有数据读进来，按文件分组
        file_data = defaultdict(list)
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                file_data[obj['litmus']].append(obj)
            except:
                pass

    # 找出每个文件的 Best Param
    for lname, records in file_data.items():
        if not records: continue
        # 找分最高的
        best_rec = max(records, key=lambda x: x['score'])
        file_best_param[lname] = tuple(best_rec['param'])

    # 检查冲突
    conflict_cnt = 0
    total_check = 0

    for vec, files in vec_to_litmus_list.items():
        if len(files) < 2: continue

        # 拿到这些文件的最优参数列表
        best_params = []
        valid_files = []
        for f in files:
            if f in file_best_param:
                best_params.append(file_best_param[f])
                valid_files.append(f)

        if len(best_params) < 2: continue

        # 看看是不是都一样
        # 如果大家的最优参数都不一样，说明这个 Vector 根本没概括性
        unique_best = set(best_params)
        total_check += 1

        if len(unique_best) > 1:
            conflict_cnt += 1
            if conflict_cnt <= 100:  # 打印前几个冲突案例
                print(f"\n[冲突案例 #{conflict_cnt}] Vector: {vec}")
                print(f"  这些文件长得一样，但脾气完全不同：")
                for i in range(min(3, len(valid_files))):
                    fname = valid_files[i]
                    print(f"    - {fname} -> 最爱参数: {best_params[i]}")

    if total_check > 0:
        print(f"\n冲突比例: {conflict_cnt}/{total_check} ({conflict_cnt / total_check:.2%})")
        if conflict_cnt / total_check > 0.5:
            print("❌ 结论: 严重冲突！大部分共用 Vector 的文件，其最优参数居然不一样。")
            print("   难怪 Train Rho 上不去，因为模型根本不知道该听谁的。")
    else:
        print("\n无法检测冲突（数据不足）。")


if __name__ == "__main__":
    analyze_vector_quality()