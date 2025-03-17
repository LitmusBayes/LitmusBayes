import json
import numpy as np
import pandas as pd

# ==========================================
# 1. 数据加载
# ==========================================

import json
import statistics

# --- 1. 文件路径配置 ---
BASELINE_FILE = '../experiment/RQ1/final_result/init/median_results.json'
PERPLE_FILE = '../experiment/perple/C910_log/median_results.json'

BAYES_FILE = '../experiment/RQ1/final_result/bayes_final/median_results.json'

# --- 2. 数据读取 ---
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

# --- 3. 变量初始化 ---
speedups_perple = []
speedups_bayes = []
raw_scores_base = []
raw_scores_perple = []
raw_scores_bayes = []

total_lat_base = 0.0
total_lat_perple = 0.0
total_lat_bayes = 0.0

valid_latency_tests = 0
print(perple_data)
# --- 4. 遍历与核心逻辑计算 ---
times = 0


keys = set(bayes_data.keys()).union(set(perple_data.keys())).union(set(baseline_data.keys()))

# for litmus, s_base in baseline_data.items():
for litmus in keys:
    s_base = baseline_data.get(litmus, 0)
    # 假设如果 LitmusBayes 中缺失，分数为0；PerpLE 缺失保留为 None 方便判定
    s_bayes = bayes_data.get(litmus, 0)
    s_perple = perple_data.get(litmus, None)


    # ==========================================
    # 逻辑一：Efficiency 计算
    # ==========================================
    raw_scores_base.append(s_base)

    # 1. PerpLE

    if s_perple is None:
        # 在 PerpLE 中没有找到，加速比认为是 1
        if s_base > 0:
            speedup_p = 1.0
            raw_scores_perple.append(s_base)  # 缺失时分数对齐 baseline
            speedups_perple.append(speedup_p)
    else:
        raw_scores_perple.append(s_perple)
        if s_base > 0:
            if s_perple <= 0:
                speedup_p = 0.0
            else:
                speedup_p = s_perple / s_base
            speedups_perple.append(speedup_p)

    # 2. LitmusBayes
    raw_scores_bayes.append(s_bayes)
    if s_base > 0:
        if s_bayes <= 0:
            speedup_b = 0
        else:
            speedup_b = s_bayes / s_base
        speedups_bayes.append(speedup_b)

    # ==========================================
    # 逻辑二：Latency 计算
    # ==========================================
    # 条件：至少有一个大于0，且不能有任何一个为-1
    has_positive = (s_base > 0) or (s_bayes > 0) or (s_perple is not None and s_perple > 0)
    if (s_perple is not None and s_perple > 0 and s_base <= 0 and s_bayes <= 0):
        print(litmus)
    has_neg_one = (s_base == -1) or (s_bayes == -1) or (s_perple == -1)
    has_neg_one = (s_bayes == -1)

    if has_positive and not has_neg_one:
        valid_latency_tests += 1

        # Baseline 延迟
        lat_base = (3.0 / s_base) if s_base > 0 else 3.0
        total_lat_base += lat_base
        times += 1

        if (s_bayes > 0 and s_base <= 0 and s_bayes <= 1):
            print(litmus, s_bayes)
        # Bayes 延迟
        lat_bayes = (3.0 / s_bayes) if s_bayes > 0 else 3.0
        total_lat_bayes += lat_bayes
        # PerpLE 延迟 (如果在 perple 找不到，直接使用 default(baseline) 时间)
        if s_perple is None:
            lat_perple = lat_base
        else:
            lat_perple = (3.0 / s_perple) if s_perple > 0 else 3.0

        total_lat_perple += lat_perple
        if s_perple is None or s_perple <= 0:
            if s_base <= 0 and s_bayes > 0:
                print(litmus)
print(times)
# --- 5. 汇总数据与打印结果 ---
print("-" * 40)
print("表 3 (Table 3) 填表数据参考:")
print("-" * 40)

# 1. Default 指标
print(f"[Default]")
print(f"  Accel. (%) (平均原始分数): {statistics.mean(raw_scores_base):.1f}")  # 若 Accel 表示平均分
print(f"  Avg. Improv.: 1.00x")
print(f"  Med. Improv.: 1.00x")
print(f"  Latency Seconds: {total_lat_base:.1f}")
print(f"  Latency Improv.: 1.00x\n")

# 2. PerpLE 指标
avg_improv_p = statistics.mean(speedups_perple)
med_improv_p = statistics.median(speedups_perple)
lat_improv_p = total_lat_base / total_lat_perple if total_lat_perple > 0 else 0
accel_ratio_p = sum(1 for x in speedups_perple if x > 1.0) / len(speedups_perple) * 100

print(f"[PerpLE]")
print(f"  Accel. (%) (平均原始分数): {statistics.mean(raw_scores_perple):.1f}")
print(f"  Accel. (%) (加速比>1占比): {accel_ratio_p:.1f}%")
print(f"  Avg. Improv.: {avg_improv_p:.2f}x")
print(f"  Med. Improv.: {med_improv_p:.2f}x")
print(f"  Latency Seconds: {total_lat_perple:.1f}")
print(f"  Latency Improv.: {lat_improv_p:.2f}x\n")

# 3. LitmusBayes 指标
avg_improv_b = statistics.mean(speedups_bayes)
med_improv_b = statistics.median(speedups_bayes)
lat_improv_b = total_lat_base / total_lat_bayes if total_lat_bayes > 0 else 0
accel_ratio_b = sum(1 for x in speedups_bayes if x > 1.0) / len(speedups_bayes) * 100

print(f"[LitmusBayes (Ours)]")
print(f"  Accel. (%) (平均原始分数): {statistics.mean(raw_scores_bayes):.1f}")
print(f"  Accel. (%) (加速比>1占比): {accel_ratio_b:.1f}%")
print(f"  Avg. Improv.: {avg_improv_b:.2f}x")
print(f"  Med. Improv.: {med_improv_b:.2f}x")
print(f"  Latency Seconds: {total_lat_bayes:.1f}")
print(f"  Latency Improv.: {lat_improv_b:.2f}x\n")

print("-" * 40)
print(f"注: 参与 Latency 计算的有效测试数量为 {valid_latency_tests} 个。")