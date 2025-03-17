import json
import os


def mutate_cycles(input_file, output_file):
    # 1. 定义目标 Fence 类型
    fence_types = [
        "Fence.r.rw", "Fence.rw.rw", "Fence.r.r", "Fence.rw.r",
        "Fence.r.w", "Fence.w.r", "Fence.rw.w", "Fence.w.rw",
        "Fence.w.w", "Fence.tso"
    ]

    # 按长度降序排序，确保优先匹配长前缀
    fence_types.sort(key=len, reverse=True)

    # 【修改点1】使用 set (集合) 来自动去重
    mutated_cycles_set = set()

    # 2. 读取输入的 JSON 文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            cycles = json.load(f)
    except Exception as e:
        print(f"读取 JSON 出错: {e}")
        return

    print(f"正在处理 {len(cycles)} 条数据...")

    count_valid_found = 0

    # 3. 遍历并处理
    for cycle_str in cycles:
        tokens = cycle_str.split()

        # 记录找到的 Fence 的 (索引, 匹配到的类型)
        found_fences = []

        for idx, token in enumerate(tokens):
            for ft in fence_types:
                # 检查 token 是否以某种 Fence 类型开头
                if token.startswith(ft):
                    found_fences.append((idx, ft))
                    break  # 找到匹配的类型后，跳过其他类型检查

        # 4. 检查条件：必须恰好存在 2 个 Fence 项
        if len(found_fences) == 2:
            new_tokens = tokens[:]  # 复制列表

            for idx, ft in found_fences:
                original_token = new_tokens[idx]
                # 替换逻辑：Fence -> po
                suffix = original_token[len(ft):]
                new_tokens[idx] = "Po" + suffix

            # 重新组合成字符串
            mutated_str = " ".join(new_tokens)

            # 【修改点2】添加到集合中（自动去重）
            mutated_cycles_set.add(mutated_str)
            count_valid_found += 1

    # 5. 保存结果
    try:
        # 【修改点3】将 set 转回 list 才能存入 JSON
        # 使用 sorted() 可以让输出结果按字母顺序排列，方便查看（如果不需要排序，直接用 list() 即可）
        final_output = sorted(list(mutated_cycles_set))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print(f"处理完成！")
        print(f"原始符合条件的条目数: {count_valid_found}")
        print(f"去重后的唯一条目数: {len(final_output)}")
        print(f"已过滤掉的重复条目: {count_valid_found - len(final_output)}")
        print(f"结果已保存至: {output_file}")

    except Exception as e:
        print(f"写入文件出错: {e}")


if __name__ == "__main__":
    input_json = "cycles.json"
    output_json = "mutated_cycles.json"

    mutate_cycles(input_json, output_json)