import json
import statistics

# --- 1. 文件路径配置 ---
BASELINE_FILE = '../experiment/RQ2/log_record_init_banana.log.validation_cache.jsonl'
# BAYES_FILE1 = '../experiment/RQ2/log_record_best_prior_final_debug_1_banana.log.validation_cache.jsonl'  # Added First Bayes File
BAYES_FILE1 = '../experiment/RQ2/final_results/prior/log_record_best_prior_4.log.validation_cache.jsonl'  # Added First Bayes File
# BAYES_FILE1 = '../experiment/RQ2/log_record_best_post_final_kmeans_10_factor_1_2_banana.log.validation_cache.jsonl'  # Added Second Bayes File
BAYES_FILE2 = '../experiment/RQ2/log_record_best_post_final_kmeans_10_factor_1_5_banana.log.validation_cache.jsonl'  # Added Second Bayes File
# BAYES_FILE2 = '../experiment/RQ2/final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_12.log.validation_cache.jsonl'  # Added Second Bayes File

# BASELINE_FILE = '../experiment/RQ2/final_results/init/aggregated_results.json'
# BAYES_FILE1 = '../experiment/RQ2/final_results/prior/aggregated_results.json'  # Added First Bayes File
# BAYES_FILE2 = '../experiment/RQ2/final_results/post/aggregated_results.json'  # Added Second Bayes File
BASELINE_FILE = '../experiment/RQ2/final_results/init/median_results.json'
BAYES_FILE1 = '../experiment/RQ2/final_results/prior/median_results.json'  # Added First Bayes File
BAYES_FILE2 = '../experiment/RQ2/final_results/post_final/median_results.json'  # Added Second Bayes File
PERPLE_FILE = "../experiment/perple/Banana_log/median_results.json"

# PERPLE_FILE = '../perple_log_scores_banana.json'

# --- 2. 数据读取 ---
baseline_data = {}
with open(BASELINE_FILE, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            baseline_data[item['litmus']] = item['score']

with open(PERPLE_FILE, 'r') as f:
    perple_data = json.load(f)

bayes_data1 = {}
with open(BAYES_FILE1, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            bayes_data1[item['litmus']] = item['score']

bayes_data2 = {}
with open(BAYES_FILE2, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            bayes_data2[item['litmus']] = item['score']

# --- 3. 变量初始化 ---
speedups_perple = []
speedups_bayes1 = []
speedups_bayes2 = []
raw_scores_base = []
raw_scores_perple = []
raw_scores_bayes1 = []
raw_scores_bayes2 = []

total_lat_base = 0.0
total_lat_perple = 0.0
total_lat_bayes1 = 0.0
total_lat_bayes2 = 0.0

valid_latency_tests = 0

# --- 4. 遍历与核心逻辑计算 ---

bayes1_yes_but2_not = 0
bayes2_yes_but1_not = 0
for litmus, s_base in baseline_data.items():
    s_bayes1 = bayes_data1.get(litmus, 0)
    s_bayes2 = bayes_data2.get(litmus, 0)
    s_perple = perple_data.get(litmus, None)

    # ==========================================
    # 逻辑一：Efficiency 计算
    # ==========================================
    raw_scores_base.append(s_base)

    # 1. PerpLE
    if s_perple is None:
        if s_base > 0:
            speedup_p = 1.0
            raw_scores_perple.append(s_base)
            speedups_perple.append(speedup_p)
    else:
        raw_scores_perple.append(s_perple)
        if s_base > 0:
            if s_perple <= 0:
                speedup_p = 0.0
            else:
                speedup_p = s_perple / s_base
            speedups_perple.append(speedup_p)

    # 2. LitmusBayes 1
    raw_scores_bayes1.append(s_bayes1)
    if s_base > 0:
        if s_bayes1 <= 0:
            speedup_b1 = 0
        else:
            speedup_b1 = s_bayes1 / s_base
        speedups_bayes1.append(speedup_b1)

    # 3. LitmusBayes 2
    raw_scores_bayes2.append(s_bayes2)
    if s_base > 0:
        if s_bayes2 <= 0:
            speedup_b2 = 0
        else:
            speedup_b2 = s_bayes2 / s_base
        speedups_bayes2.append(speedup_b2)

    # ==========================================
    # 逻辑二：Latency 计算
    # ==========================================
    has_positive = (s_base > 0) or (s_bayes1 > 0) or (s_bayes2 > 0) or (s_perple is not None and s_perple > 0)

    # Consider -1 as an invalid test for latency computation
    has_neg_one = (s_base == -1) or (s_bayes1 == -1) or (s_bayes2 == -1) or (s_perple == -1)

    if has_positive and not has_neg_one:

        s_base = s_base if s_base != None else 0
        s_bayes1 = s_bayes1 if s_bayes1 != None else 0
        s_bayes2 = s_bayes2 if s_bayes2 != None else 0

        min_value = max(s_base, s_bayes1, s_bayes2)
        if s_perple != None:
            min_value = max(min_value, s_perple)
        if s_base > 0:
            min_value = min(min_value, s_base)
        if s_bayes1 > 0:
            min_value = min(min_value, s_bayes1)
        if s_bayes2 > 0:
            min_value = min(min_value, s_bayes2)
        if s_perple!= None and s_perple > 0:
            min_value = min(min_value, s_perple)
        #
        if s_perple == None or s_perple <= 0:
            if s_base <= 0 and s_bayes1 > 0 and s_bayes2 <= 0:
                print(litmus)
            #     continue
            # if s_base <= 0 and s_bayes1 <= 0 and s_bayes2 > 0:
            #     continue
        p_value = 3.0

        if s_base <= 0:
            if s_perple == None or s_perple <= 0:
                if (s_bayes2 > 0 and s_bayes1 <= 0) or (s_bayes2 <= 0 and s_bayes1 > 0):
                    continue
        # if min_value < 1:
        #     p_value = p_value / min_value

        # Baseline 延迟

        lat_base = (3.0 / s_base) if s_base > 0 else p_value
        total_lat_base += lat_base

        # Bayes 1 延迟
        lat_bayes1 = (3.0 / s_bayes1) if s_bayes1 > 0 else p_value
        total_lat_bayes1 += lat_bayes1

        # Bayes 2 延迟
        lat_bayes2 = (3.0 / s_bayes2) if s_bayes2 > 0 else p_value
        total_lat_bayes2 += lat_bayes2

        # PerpLE 延迟
        if s_perple is None:
            lat_perple = lat_base
        else:
            lat_perple = (3.0 / s_perple) if s_perple > 0 else p_value

        total_lat_perple += lat_perple
        if s_bayes1 > 0 and s_bayes2 <=0:
            bayes1_yes_but2_not += 1
        if s_bayes2 > 0 and s_bayes1 <=0:
            bayes2_yes_but1_not += 1
        valid_latency_tests += 1

# --- 5. 汇总数据与打印结果 ---
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
avg_improv_b = statistics.mean(speedups_bayes1)
med_improv_b = statistics.median(speedups_bayes1)
lat_improv_b = total_lat_base / total_lat_bayes1 if total_lat_bayes1 > 0 else 0
accel_ratio_b = sum(1 for x in speedups_bayes1 if x > 1.0) / len(speedups_bayes1) * 100

print(f"[LitmusBayes (non trans)]")
print(f"  Accel. (%) (平均原始分数): {statistics.mean(raw_scores_bayes1):.1f}")
print(f"  Accel. (%) (加速比>1占比): {accel_ratio_b:.1f}%")
print(f"  Avg. Improv.: {avg_improv_b:.2f}x")
print(f"  Med. Improv.: {med_improv_b:.2f}x")
print(f"  Latency Seconds: {total_lat_bayes1:.1f}")
print(f"  Latency Improv.: {lat_improv_b:.2f}x\n")

# 3. LitmusBayes 指标
avg_improv_b = statistics.mean(speedups_bayes2)
med_improv_b = statistics.median(speedups_bayes2)
lat_improv_b = total_lat_base / total_lat_bayes2 if total_lat_bayes2 > 0 else 0
accel_ratio_b = sum(1 for x in speedups_bayes2 if x > 1.0) / len(speedups_bayes2) * 100

print(f"[LitmusBayes (ours)]")
print(f"  Accel. (%) (平均原始分数): {statistics.mean(raw_scores_bayes2):.1f}")
print(f"  Accel. (%) (加速比>1占比): {accel_ratio_b:.1f}%")
print(f"  Avg. Improv.: {avg_improv_b:.2f}x")
print(f"  Med. Improv.: {med_improv_b:.2f}x")
print(f"  Latency Seconds: {total_lat_bayes2:.1f}")
print(f"  Latency Improv.: {lat_improv_b:.2f}x\n")

print("-" * 40)
print(f"注: 参与 Latency 计算的有效测试数量为 {valid_latency_tests} 个。")
print("bayes1_yes_but2_not", bayes1_yes_but2_not)
print("bayes2_yes_but1_not", bayes2_yes_but1_not)