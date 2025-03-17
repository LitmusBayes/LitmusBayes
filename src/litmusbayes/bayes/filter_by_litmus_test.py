import json
import os

# ================= 配置输入 =================

import json
import os


def filter_litmus_data(input_file, target_litmus_name, output_file=None):
    """
    从 JSONL 文件中过滤出指定 litmus test 的所有条目。

    :param input_file: 原始数据文件路径
    :param target_litmus_name: 想要筛选的 litmus 名字 (字符串精确匹配)
    :param output_file: (可选) 结果保存路径，如果不填则自动生成
    """

    # 如果没有指定输出文件名，自动生成一个
    if output_file is None:
        # 替换文件名中的特殊字符，防止路径错误
        safe_name = target_litmus_name.replace("+", "_").replace("/", "-")
        output_file = f"filtered_{safe_name}.jsonl"

    print(f"正在从 [{input_file}] 中筛选: {target_litmus_name} ...")

    found_count = 0
    results = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
                open(output_file, 'w', encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # 核心判断逻辑
                    if data.get("litmus") == target_litmus_name:
                        results.append(data)
                        # 立即写入输出文件
                        f_out.write(json.dumps(data) + "\n")
                        found_count += 1

                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行不是有效的 JSON，已跳过。")
                    continue

        print("-" * 30)
        print(f"筛选完成！")
        print(f"共找到 {found_count} 条数据。")
        print(f"结果已保存至: {output_file}")

        return results

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return []


# ================= 使用示例 =================
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"

# 假设这是你的数据文件路径
data_file_path = stat_log_base + ".cache4_norm.jsonl"
# 假设这是你的重复分组文件路径
groups_file_path = "identical_vector_groups.json"
# 输出路径
output_file_path = stat_log_base + ".cache4_norm_filter_same.jsonl"

if __name__ == "__main__":
    # 1. 你的原始文件路径 (请修改这里)
    my_data_file = data_file_path  # 假设你的文件名叫 data.jsonl

    # 2. 你想找的 Litmus 名字 (请修改这里)
    # 例如你想找 prompt 中最后一条那个分数很高的
    target_name = "SB+po+pos-po-addrs"

    # 3. 运行过滤
    # 如果文件不存在，你可以先创建一个假文件测试
    if not os.path.exists(my_data_file):
        print(f"提示：找不到 {my_data_file}，将为你创建一个包含示例数据的临时文件用于测试...")
        sample_data = [
            {"litmus": "MP+rfi-ctrl+data-rfi-ctrlfenceis", "param": [0, 1, 1], "score": 0.15},
            {"litmus": "2+2W+rfi-addrs", "param": [1, 5, 1], "score": 36.8},
            {"litmus": "2+2W+rfi-addrs", "param": [0, 0, 0], "score": 12.5},  # 另一条同名的
            {"litmus": "SB+rfi-pos", "param": [1, 3, 0], "score": 0.02}
        ]
        with open(my_data_file, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

    # 执行函数
    filter_litmus_data(my_data_file, target_name)