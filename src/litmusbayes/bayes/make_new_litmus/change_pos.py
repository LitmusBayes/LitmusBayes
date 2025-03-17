import json
import os
from itertools import combinations


def mutate_pos_cycles(input_file, output_file):
    # 使用集合自动去重
    mutated_cycles_set = set()

    # 1. 读取输入文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            cycles = json.load(f)
    except Exception as e:
        print(f"读取 JSON 出错: {e}")
        return

    print(f"正在扫描 {len(cycles)} 条原始数据...")

    source_count = 0  # 记录有多少条原始数据产生了新变异

    # 2. 遍历每一条 cycle
    for cycle_str in cycles:
        tokens = cycle_str.split()

        # 找出所有包含 "Pos" 的 token 的索引
        # 例如: tokens[1] 是 "PosWW", tokens[4] 是 "PosRRXP" -> indices = [1, 4]
        pos_indices = [i for i, token in enumerate(tokens) if "Pos" in token]

        # 只有当 Pos 项至少有 2 个时，才能进行“两两一对”的转换
        if len(pos_indices) >= 2:
            generated_for_this_cycle = False

            # 使用 combinations 生成所有两两组合
            # 如果 indices 是 [1, 4, 6] (即3个Pos)，这里会生成 (1,4), (1,6), (4,6)
            for idx1, idx2 in combinations(pos_indices, 2):
                new_tokens = tokens[:]  # 复制当前 token 列表

                # 执行替换: Pos -> Pod
                # 注意：这里直接替换字符串中的 "Pos" 为 "Pod"
                new_tokens[idx1] = new_tokens[idx1].replace("Pos", "Pod")
                new_tokens[idx2] = new_tokens[idx2].replace("Pos", "Pod")

                # 重新组合并添加到结果集
                new_cycle_str = " ".join(new_tokens)
                mutated_cycles_set.add(new_cycle_str)
                generated_for_this_cycle = True

            if generated_for_this_cycle:
                source_count += 1

    # 3. 保存结果
    try:
        # 排序后转换为列表
        final_output = sorted(list(mutated_cycles_set))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print(f"处理完成！")
        print(f"产生变异的原始条目数: {source_count}")
        print(f"生成的唯一新条目总数: {len(final_output)}")
        print(f"结果已保存至: {output_file}")

    except Exception as e:
        print(f"写入文件出错: {e}")


if __name__ == "__main__":
    # 输入文件（可以是原始的 cycles.json，也可以是上一步生成的）
    input_json = "cycles.json"
    # 输出文件
    output_json = "pos_mutated_cycles.json"

    mutate_pos_cycles(input_json, output_json)