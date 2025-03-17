import json
from collections import defaultdict


def process_litmus_file(input_file, output_file):
    # 1. 创建一个字典用于分组存储数据
    # key 是 litmus 的名字, value 是该名字对应的列表
    grouped_data = defaultdict(list)

    print(f"正在读取文件: {input_file} ...")

    # 2. 逐行读取并解析
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)
                name = data.get("litmus")
                # 将数据加入对应的组
                grouped_data[name].append(data)
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line}")

    print("正在排序和整理...")

    # 3. 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 遍历每一个 litmus 组
        for name, items in grouped_data.items():
            # 核心步骤：按照 score 从大到小排序 (reverse=True)
            # 如果 score 相同，保持原顺序
            items.sort(key=lambda x: x.get("score", 0), reverse=True)

            # 将排序好的每条数据写入新文件
            for item in items:
                # ensure_ascii=False 保证中文等字符正常显示（如果有的话）
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"完成！已保存到: {output_file}")


# --- 配置区 ---
# 请在这里修改您的输入和输出文件名

stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"
input_filename = stat_log_base + ".cache_sum_70_no.jsonl"
output_filename = stat_log_base + ".cache_sum_70_no_sort.jsonl"  # 输出的新文件名
if __name__ == "__main__":
    process_litmus_file(input_filename, output_filename)
