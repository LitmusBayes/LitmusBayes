import json
import os


def merge_and_deduplicate(files_list, output_file):
    """
    读取文件列表中的所有 JSON，合并并去重，最后保存。
    """
    # 使用 set (集合) 自动去重
    merged_set = set()

    print(f"准备合并以下 {len(files_list)} 个文件: {files_list}")
    print("-" * 40)

    for file_path in files_list:
        if not os.path.exists(file_path):
            print(f"[跳过] 文件不存在: {file_path}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    # 记录合并前的数量
                    count_before = len(merged_set)

                    # 将列表中的条目更新到集合中
                    merged_set.update(data)

                    # 计算新增的唯一条目数
                    added_count = len(merged_set) - count_before
                    print(f"[成功] 读取 {file_path}: \t发现 {len(data)} 条，新增唯一条目 {added_count} 条")
                else:
                    print(f"[警告] 文件 {file_path} 的内容不是列表格式，已跳过。")

        except Exception as e:
            print(f"[错误] 读取 {file_path} 时发生异常: {e}")

    # 结果保存
    print("-" * 40)
    try:
        # 转回列表并排序，保持输出整洁
        final_list = sorted(list(merged_set))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, indent=4, ensure_ascii=False)

        print(f"合并完成！")
        print(f"最终去重后的总条目数: {len(final_list)}")
        print(f"文件已保存至: {output_file}")

    except Exception as e:
        print(f"[错误] 写入结果文件时出错: {e}")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 在这里列出你需要合并的三个文件名（或者是之前的原始文件）
    input_files = [
        "mutated_cycles.json",  # 1. Fence 变异文件
        "pos_mutated_cycles.json",  # 2. Pos 变异文件
        # "rw_wr_mutated_cycles.json"  # 3. RW/WR 变异文件
        # "cycles.json"              # 如果你想把最原始的也合进去，可以把这行取消注释
    ]

    output_filename = "all_cycles_merged.json"

    # --- 运行合并 ---
    merge_and_deduplicate(input_files, output_filename)