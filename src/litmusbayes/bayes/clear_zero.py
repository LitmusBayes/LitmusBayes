import json
from collections import defaultdict


def filter_zero_litmus(input_path, output_path):
    print(f"正在处理文件: {input_path} ...")

    # 1. 第一遍扫描：记录每个 Litmus Test 是否出现过非 0 分数
    # 使用集合来存储"有效的"（即至少有过一次非0分）的 Litmus 名称
    valid_litmus_names = set()

    # 同时我们需要把数据暂存起来（因为文件不大，18000行可以直接存内存）
    all_data = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            try:
                item = json.loads(line)
                all_data.append(item)

                name = item['litmus']
                score = item['score']

                # 只要发现一次分数不为0，就标记这个 Litmus 为"有效"
                if score != 0:
                    valid_litmus_names.add(name)
            except json.JSONDecodeError:
                print(f"跳过无效行: {line}")

    # 2. 第二遍扫描：写入过滤后的数据
    with open(output_path, 'w') as f:
        kept_count = 0
        removed_count = 0

        for item in all_data:
            name = item['litmus']

            # 只有当这个名字在"有效列表"里，我们才保留这条数据
            if name in valid_litmus_names:
                f.write(json.dumps(item) + "\n")
                kept_count += 1
            else:
                removed_count += 1

    # 3. 打印统计信息
    total_litmus_count = len(set(d['litmus'] for d in all_data))
    valid_litmus_count = len(valid_litmus_names)
    removed_litmus_count = total_litmus_count - valid_litmus_count

    print("-" * 40)
    print("清洗完成！统计结果如下：")
    print(f"原始 Litmus Test 总数: {total_litmus_count}")
    print(f"被完全移除的 Test 数:  {removed_litmus_count} (所有采样均为0分)")
    print(f"保留的有效 Test 数:    {valid_litmus_count}")
    print("-" * 40)
    print(f"原始数据行数: {len(all_data)}")
    print(f"移除数据行数: {removed_count}")
    print(f"保留数据行数: {kept_count}")
    print(f"新文件已保存至: {output_path}")


# ================= 配置 =================
if __name__ == "__main__":
    # 请修改这里的路径为你实际的文件路径
    input_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum.jsonl"
    output_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes_cleaned.jsonl"

    filter_zero_litmus(input_file, output_file)