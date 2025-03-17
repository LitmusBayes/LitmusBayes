import json


def process_and_normalize(data_file, ref_file, output_file):
    """
    使用 ref_file 中的分数作为分母，对 data_file 中的分数进行归一化。
    """

    # --- 步骤 1: 读取第二个文件 (参考分数) ---
    print(f"正在加载参考文件: {ref_file} ...")
    reference_map = {}
    try:
        with open(ref_file, 'r', encoding='utf-8') as f:
            reference_map = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到参考文件 {ref_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: 参考文件 {ref_file} 格式不正确")
        return

    print(f"参考文件加载完毕，包含 {len(reference_map)} 个基准测试的分数。")

    # --- 步骤 2: 处理第一个文件并写入结果 ---
    print(f"正在处理数据文件: {data_file} ...")

    processed_count = 0
    skipped_count = 0

    try:
        with open(data_file, 'r', encoding='utf-8') as f_in, \
                open(output_file, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                try:
                    # 解析当前行的数据
                    record = json.loads(line)
                    litmus_name = record.get("litmus")
                    original_score = record.get("score")

                    # 检查参考文件中是否有对应的 litmus name
                    if litmus_name in reference_map:
                        denominator = reference_map[litmus_name]

                        # 防止除以 0
                        if denominator == 0:
                            print(f"警告: {litmus_name} 的基准分数为 0，跳过此行。")
                            skipped_count += 1
                            continue

                        # === 核心计算 ===
                        new_score = original_score / denominator

                        # 更新分数
                        record["score"] = new_score

                        # 写入新文件
                        f_out.write(json.dumps(record) + "\n")
                        processed_count += 1
                    else:
                        # 如果在第二个文件中找不到对应的名字，直接跳过
                        # print(f"跳过: {litmus_name} 在参考文件中不存在。")
                        skipped_count += 1

                except json.JSONDecodeError:
                    print("警告: 数据文件中有一行无法解析。")

    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_file}")
        return

    print("-" * 30)
    print(f"处理完成！")
    print(f"成功转换并保存: {processed_count} 行")
    print(f"跳过 (未找到匹配或基准分为0): {skipped_count} 行")
    print(f"结果已保存至: {output_file}")


# --- 配置区域 ---

# 你的第一个文件 (JSON Lines 格式，待处理数据)
input_data_file = 'log/cache_0.jsonl'

# 你的第二个文件 (标准 JSON 格式，包含基准分)
# 也就是你刚才生成的 filtered_scores.json
input_ref_file = 'log/baseline_scores_0.json'

# 输出文件
output_result_file = 'log/cache_norm_0.jsonl'

# --- 执行 ---
if __name__ == "__main__":
    # 请确保目录下有对应的文件，或者修改上面的文件名路径
    process_and_normalize(input_data_file, input_ref_file, output_result_file)