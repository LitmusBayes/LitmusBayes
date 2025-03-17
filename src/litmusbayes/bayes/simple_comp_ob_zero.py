import json
import re
import statistics


def parse_log1(content):
    """
    解析日志1 (Text格式) - 保持不变
    """
    data = {}
    current_test = None
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.endswith(" result:"):
            current_test = line.replace(" result:", "").strip()
        elif line.startswith("all_allow_litmus:") and current_test:
            try:
                parts = line.split(":")[1].split(",")
                if len(parts) >= 2:
                    score = float(parts[1].strip())
                    data[current_test] = score
            except (ValueError, IndexError):
                pass
    return data


def parse_log2_with_params(content):
    """
    [修改] 解析日志2 (JSON Lines格式)
    逻辑: 读取每一行JSON，保留分数最大的那一行的 score 和 param
    """
    data = {}  # 结构: { "litmus_name": {"score": 123.0, "param": [...]} }

    lines = content.strip().split('\n')
    for line in lines:
        if not line.strip(): continue
        try:
            entry = json.loads(line)
            name = entry.get("litmus")
            # print(entry) # 调试时可打开

            val = entry.get("score", 0)
            if val is None: continue
            score = float(val)
            param = entry.get("param", [])  # 获取参数

            if name:
                # 如果该测试已存在，比较分数
                if name in data:
                    if score > data[name]["score"]:
                        data[name] = {"score": score, "param": param}
                else:
                    data[name] = {"score": score, "param": param}
        except json.JSONDecodeError:
            continue

    return data


def parse_reference_log(content):
    """
    [新增] 解析参考日志 (JSON Lines格式)
    逻辑: 创建一个查找表，Key为 (litmus_name, tuple(param))，Value为 score
    注意: param 需要转为 tuple 才能作为字典的 Key
    """
    ref_data = {}
    lines = content.strip().split('\n')
    for line in lines:
        if not line.strip(): continue
        try:
            entry = json.loads(line)
            name = entry.get("litmus")
            param = entry.get("param")
            val = entry.get("score")

            if name and param is not None and val is not None:
                # 将 param 列表转为 tuple 以便哈希
                key = (name, tuple(param))
                ref_data[key] = float(val)
        except json.JSONDecodeError:
            continue
    return ref_data


def compare_logs(log1_data, log2_data, ref_data):
    """
    [修改] 比较逻辑，增加了 ref_data 用于查找
    """
    count = 0
    details = []
    summary = []
    zero1 = 0
    zero2 = 0
    all_zero = 0
    not_zero = 0
    not_pass = []
    times1 = 0
    times2 = 0
    exp = 0

    # 遍历 Log2
    for test_name, item2 in log2_data.items():
        # 解包 score 和 param
        score2 = item2["score"]
        param2 = item2["param"]

        if test_name in log1_data:
            score1 = log1_data[test_name]

            # --- 比较逻辑 ---
            if score1 == score2 and score1 == 0:
                all_zero += 1
            elif score2 > score1:
                count += 1
                details.append({
                    "test": test_name,
                    "log1_score": score1,
                    "log2_max_score": score2
                })
                if score1 != 0:
                    summary.append(score2 / score1)
                    times1 += 3 / score1
                    times2 += 3 / score2
                    not_zero += 1
                else:
                    zero1 += 1
            elif score1 > score2 and score2 != 0 and score2 != -1:
                not_pass.append((test_name, score1, score2))
                times2 += 3 / score2
                times1 += 3 / score1
                not_zero += 1
            elif score2 == score1 and score2 != 0 and score2 != -1:
                times2 += 3 / score2
                times1 += 3 / score1
                not_zero += 1
            elif score2 == -1 or score2 == 0:
                if score2 == -1:
                    print("zero2 -1", test_name)
                    exp += 1
                else:
                    print("zero2", test_name)
                not_zero += 1
                zero2 += 1

                # ==========================================
                # [新增功能] 在 Reference Log 中查找
                # ==========================================
                # 构造查找键: (name, tuple(param))
                search_key = (test_name, tuple(param2))
                if search_key in ref_data:
                    ref_score = ref_data[search_key]
                    print(
                        f"[Reference Found] Test: {test_name}, Score2: {score2}, Params: {param2} -> Ref Score: {ref_score}")
                # else:
                #     print(f"[Reference Not Found] Test: {test_name} with params {param2}")
                # ==========================================

    if summary:
        mean_val = sum(summary) / len(summary)
        median_val = statistics.median(summary)
    else:
        mean_val = 0
        median_val = 0
    return count, details, mean_val, median_val, zero1, zero2, all_zero, not_zero, not_pass, times1, times2, exp


# ==========================================
# 主程序执行部分
# ==========================================

path_log1 = '/home/whq/Desktop/code_list/perple_test/log_C910/log.txt'
path_log2 = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_for_best9.jsonl'
# 请替换为你的第三个文件路径（包含要查找的数据）
path_log3 = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache4_norm.jsonl'

# 1. 读取 Log1
with open(path_log1, 'r') as f:
    log1_raw = f.read()
data1 = parse_log1(log1_raw)

# 2. 读取 Log2 (现在包含 param)
with open(path_log2, 'r') as f:
    log2_raw = f.read()
data2 = parse_log2_with_params(log2_raw)

# 3. [新增] 读取 Reference Log (Log3)
# 假设 Log3 也是 jsonl 格式
try:
    with open(path_log3, 'r') as f:
        log3_raw = f.read()
    data3_ref = parse_reference_log(log3_raw)
    print(f"Reference log loaded, contains {len(data3_ref)} entries.")
except FileNotFoundError:
    print("Warning: Reference log file not found.")
    data3_ref = {}

# 4. 执行比较
result_count, result_details, mean_val, median_val, zero1, zero2, all_zero, not_zero, not_pass, times1, times2, exp = compare_logs(
    data1, data2, data3_ref)

print(f"--- 解析结果 ---")
# print(f"Log1 数据: {json.dumps(data1, indent=2)}")
# data2 现在包含param，打印可能比较长，建议只打印统计信息
# print(f"Log2 数据: {json.dumps(data2, indent=2)}")

print(f"\n--- 比较结果 ---")
print(f"Log2 严格大于 Log1 的测试数量: {result_count}")

# ... (后续打印逻辑保持不变)
if result_count > 0:
    print("详细列表 (前10个):")  # 避免刷屏只打印前几个
    for item in result_details[:10]:
        print(f"Test: {item['test']} | Log1: {item['log1_score']} < Log2: {item['log2_max_score']}")

print(f"not pass:{len(not_pass)}")
for litmus_name, score1, score2 in not_pass:
    print(f"litmus_name:{litmus_name}, score1:{score1}, score2:{score2}")

print(f"result_count: {result_count}")
print(f"mean_val: {mean_val}")
print(f"median_val: {median_val}")
print(f"zero1: {zero1}")
print(f"zero2: {zero2}")
print(f"not_zero: {not_zero}")
print(f"all_zero: {all_zero}")
print(f"times1:{times1}")
print(f"times2:{times2}")
if times2 != 0:
    print(f"times1/times2:{times1 / times2}")
print(f"exp:{exp}")