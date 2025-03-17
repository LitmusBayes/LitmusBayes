import json
import os


def parse_log1(content):
    """
    解析日志1 (Text格式)
    构建基准字典: { "litmus_name": score1 }
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


def process_log2_stream(log2_path, data1, output_path):
    """
    流式读取 Log2，计算新分数并逐行写入新文件
    """
    count = 0
    skipped = 0

    # 打开输入和输出文件
    with open(log2_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            if not line.strip():
                continue

            try:
                # 1. 解析原始行
                entry = json.loads(line)
                litmus_name = entry.get("litmus")

                # 2. 检查 log1 中是否有这个测试的基准分
                if litmus_name in data1:
                    score1 = data1[litmus_name]  # 分母 (基准)
                    score2 = float(entry.get("score", 0))  # 分子 (当前)

                    # 3. 计算新分数 (score2 / score1)
                    final_score = 0.0

                    if score1 == 0:
                        if score2 == 0:
                            final_score = 1.0  # 0/0 -> 1
                        else:
                            final_score = 100.0  # N/0 -> 100
                    else:
                        final_score = score2 / score1

                    # 4. 修改 entry 中的 score
                    entry["score"] = final_score

                    # 5. 写入新文件
                    fout.write(json.dumps(entry) + '\n')
                    count += 1
                else:
                    # 如果 log1 里没有这个测试，跳过或按需处理
                    # 这里选择跳过，因为没有基准分无法做除法
                    print(litmus_name)
                    skipped += 1

            except json.JSONDecodeError:
                continue

    print(f"处理完成！")
    print(f"生成文件: {output_path}")
    print(f"成功转换行数: {count}")
    print(f"跳过行数 (Log1中不存在): {skipped}")


# ==========================================
# 主程序入口
# ==========================================

file_path1 = '/home/whq/Desktop/code_list/perple_test/log_C910/log.txt'
file_path2 = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no.jsonl'
output_path = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_norm_gt_0.jsonl'

try:
    # 1. 先把 Log1 全部读入内存做成字典
    print("正在解析 Log1...")
    with open(file_path1, 'r') as f:
        data1 = parse_log1(f.read())
    print(f"Log1 加载完毕，包含 {len(data1)} 个基准测试用例。")

    # 2. 流式处理 Log2 并生成文件
    print("正在处理 Log2 并生成新文件...")
    process_log2_stream(file_path2, data1, output_path)

except FileNotFoundError as e:
    print(f"错误: 找不到文件 - {e}")