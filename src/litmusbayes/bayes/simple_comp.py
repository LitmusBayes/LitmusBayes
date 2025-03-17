
import json
import re
import statistics


def parse_log1(content):
    """
    解析日志1 (Text格式)
    逻辑: 找到测试名称行，然后找到该测试下的 all_allow_litmus 行提取分数
    """
    data = {}
    current_test = None

    # 按行处理
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()

        # 1. 识别测试名称行 (以 result: 结尾)
        if line.endswith(" result:"):
            # 移除 " result:" 获取测试名
            current_test = line.replace(" result:", "").strip()

        # 2. 识别目标分数行 (all_allow_litmus)
        elif line.startswith("all_allow_litmus:") and current_test:
            # 格式: all_allow_litmus: count, score
            # 分割字符串提取数字
            try:
                parts = line.split(":")[1].split(",")
                if len(parts) >= 2:
                    score = float(parts[1].strip())
                    data[current_test] = score
            except (ValueError, IndexError):
                pass

    return data

def parse_log2(content):
    """
    解析日志2 (JSON Lines格式)
    逻辑: 读取每一行JSON，对于相同的 litmus，只保留最大的 score
    """
    data = {}

    lines = content.strip().split('\n')
    for line in lines:
        if not line.strip(): continue
        try:
            entry = json.loads(line)
            name = entry.get("litmus")
            print(entry)
            if entry.get("score", 0) is None:
                continue
            score = float(entry.get("score", 0))

            if name:
                # 如果该测试已存在，取当前分数与已知最大值的较大者
                if name in data:
                    data[name] = max(data[name], score)
                else:
                    data[name] = score
        except json.JSONDecodeError:
            continue

    return data

def compare_logs(log1_data, log2_data):
    """
    比较两个数据集
    返回: 计数, 详细列表
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
    # 遍历 Log2 中所有的测试 (因为我们要找的是 Log2 > Log1)
    for test_name, score2 in log2_data.items():
        # 只有当 Log1 中也有这个测试时才比较
        if test_name in log1_data:
            score1 = log1_data[test_name]

            # 严格大于
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
                    summary.append(score2/score1)
                    times1 = times1 + 3 / score1
                    times2 = times2 + 3 / score2
                    not_zero += 1
                else:
                    zero1 += 1
            elif score1 > score2 and score2 != 0 and score2 != -1:
                not_pass.append((test_name, score1, score2))
                times2 = times2 + 3 / score2
                times1 = times1 + 3 / score1
                not_zero += 1
            elif score2 == score1 and score2 != 0 and score2 != -1:
                times2 = times2 + 3 / score2
                times1 = times1 + 3 / score1
                not_zero += 1
            elif score2 == -1 or score2 == 0:
                if score2 == -1:
                    exp += 1
                not_zero += 1
                zero2 += 1
                print("zero2", test_name)
    if summary:

        mean_val = sum(summary) / len(summary)
        median_val = statistics.median(summary)  # 计算中位数
    else:
        mean_val = 0
        median_val = 0
    return count, details, mean_val, median_val, zero1, zero2, all_zero, not_zero, not_pass, times1, times2, exp

# ==========================================
# 这里填入你的数据进行测试
# ==========================================


# 读取文件内容
# with open('/home/whq/Desktop/code_list/perple_test/log_C910/log.txt', 'r') as f:
with open('/home/whq/Desktop/code_list/perple_test/log_C910/log.txt', 'r') as f:

    log1_raw = f.read()

# with open('/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no.jsonl', 'r') as f:
with open('/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes_banana.log.cache.jsonl',
              'r') as f:
    log2_raw = f.read()

# 剩下的代码调用 parse_log1, parse_log2 和 compare_logs 保持不变
# 执行解析
data1 = parse_log1(log1_raw)
data2 = parse_log2(log2_raw)

# 执行比较
result_count, result_details, mean_val, median_val, zero1, zero2, all_zero, not_zero, not_pass, times1, times2, exp = compare_logs(data1, data2)

print(f"--- 解析结果 ---")
print(f"Log1 数据: {json.dumps(data1, indent=2)}")
print(f"Log2 数据 (取最大值后): {json.dumps(data2, indent=2)}")
print(f"\n--- 比较结果 ---")
print(f"Log2 严格大于 Log1 的测试数量: {result_count}")

if result_count > 0:
    print("详细列表:")
    for item in result_details:
        print(f"Test: {item['test']} | Log1: {item['log1_score']} < Log2: {item['log2_max_score']}")
else:
    print("(基于你提供的片段，没有发现 Log2 分数大于 Log1 的情况)")

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
print(f"times1/times2:{times1/times2}")
print(f"exp:{exp}")