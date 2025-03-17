import json
import os


def parse_log1(content):
    """解析日志1 (Text格式) 构建基准字典"""
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


def process_log2_with_filter(log2_path, data1, output_path, skip_log_path):
    """
    带预过滤的处理函数：
    1. 过滤 score1 == 0 的 litmus
    2. 过滤 所有 score2 均为 0 的 litmus
    """
    # 用于统计每个 litmus 的 score2 之和
    litmus_score2_sum = {}
    # 记录黑名单原因
    black_list_reasons = {}

    # --- 第一阶段：扫描统计 ---
    print("正在进行第一轮扫描（预统计）...")
    with open(log2_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                name = entry.get("litmus")
                s2 = float(entry.get("score", 0))

                if name not in litmus_score2_sum:
                    litmus_score2_sum[name] = 0.0
                litmus_score2_sum[name] += s2
            except:
                continue

    # --- 第二阶段：构建过滤名单 ---
    filtered_names = set()

    # 1. 检查 Log1 中不存在或 score1 为 0
    for name in litmus_score2_sum.keys():
        if name not in data1:
            filtered_names.add(name)
            black_list_reasons[name] = "不在 Log1 基准文件中"
        elif data1[name] == 0:
            filtered_names.add(name)
            black_list_reasons[name] = "Log1 基准分(score1) 为 0"

    # 2. 检查 score2 累加和是否为 0
    for name, total_s2 in litmus_score2_sum.items():
        if name not in filtered_names and total_s2 == 0:
            filtered_names.add(name)
            black_list_reasons[name] = "该测试所有条目的 score2 均为 0"

    # --- 第三阶段：执行写入 ---
    print(f"正在进行第二轮扫描（正式写入），将跳过 {len(filtered_names)} 个测试用例...")
    count = 0

    with open(log2_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            try:
                entry = json.loads(line)
                name = entry.get("litmus")

                if name in filtered_names:
                    continue

                # 计算归一化分数
                score1 = data1[name]
                score2 = float(entry.get("score", 0))
                entry["score"] = score2 / score1

                fout.write(json.dumps(entry) + '\n')
                count += 1
            except:
                continue

    # --- 第四阶段：保存跳过名单 Log ---
    with open(skip_log_path, 'w', encoding='utf-8') as flog:
        flog.write(f"{'Litmus Name':<50} | {'Skip Reason'}\n")
        flog.write("-" * 80 + "\n")
        for name in sorted(filtered_names):
            flog.write(f"{name:<50} | {black_list_reasons.get(name, 'Unknown')}\n")

    print(f"\n处理完成！")
    print(f"有效数据行数: {count}")
    print(f"已过滤的 Litmus 种类数: {len(filtered_names)}")
    print(f"过滤详情已写入: {skip_log_path}")


# ==========================================
# 主程序入口
# ==========================================
file_path1 = '/home/whq/Desktop/code_list/perple_test/log_C910/log.txt'
file_path2 = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache4.jsonl'
output_path = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache4_norm.jsonl'
skip_log_path = '/home/whq/Desktop/code_list/perple_test/bayes_stat/filtered_litmus_log.txt'

try:
    print("正在解析 Log1...")
    with open(file_path1, 'r') as f:
        data1 = parse_log1(f.read())

    process_log2_with_filter(file_path2, data1, output_path, skip_log_path)

except Exception as e:
    print(f"运行出错: {e}")