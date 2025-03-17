import json
import os


def mutate_rw_wr_cycles(input_file, output_file):
    mutated_cycles_set = set()

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
    source_count = 0

    for cycle_str in cycles:
        tokens = cycle_str.split()
        n = len(tokens)
        generated_for_this_cycle = False

        # 遍历每一个 token
        for i in range(n):
            token = tokens[i]

            # --- 情况 1: 处理 RW -> WR ---
            if "RW" in token:
                new_tokens = tokens[:]  # 复制列表

                # 1. 修改当前项
                new_tokens[i] = new_tokens[i].replace("RW", "WR")

                # 2. 修改前一项 (Prev)
                prev_idx = (i - 1) % n
                prev_token = new_tokens[prev_idx]

                if "Rf" in prev_token:
                    # 如果含有 Rf，改为 Co (保留后缀如 Rfe -> Coe)
                    new_tokens[prev_idx] = prev_token.replace("Rf", "Co")
                elif "RR" in prev_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[prev_idx] = prev_token.replace("RR", "RW")
                elif "WR" in prev_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[prev_idx] = prev_token.replace("WR", "WW")
                else:
                    continue
                # 3. 修改后一项 (Next)
                next_idx = (i + 1) % n
                next_token = new_tokens[next_idx]

                if "Co" in next_token:
                    # 如果含有 Co，改为 Rf
                    new_tokens[next_idx] = next_token.replace("Co", "Fr")
                elif "Ws" in next_token:
                    new_tokens[next_idx] = next_token.replace("Ws", "Fr")
                elif "WW" in next_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[next_idx] = next_token.replace("WW", "RW")
                elif "WR" in next_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[next_idx] = next_token.replace("WR", "RR")
                else:
                    continue
                print("from : ", cycle_str)
                print("to : ", " ".join(new_tokens))
                mutated_cycles_set.add(" ".join(new_tokens))
                generated_for_this_cycle = True

            # --- 情况 2: 处理 WR -> RW (逻辑反转) ---
            if "WR" in token:
                new_tokens = tokens[:]  # 复制列表

                # 1. 修改当前项
                new_tokens[i] = new_tokens[i].replace("WR", "RW")

                # 2. 修改前一项 (Prev) - 逻辑反转
                # RW->WR时是 Rf->Co; 这里 WR->RW 则 Co->Rf
                prev_idx = (i - 1) % n
                prev_token = new_tokens[prev_idx]

                if "Co" in prev_token:
                    new_tokens[prev_idx] = prev_token.replace("Co", "Rf")
                elif "Ws" in prev_token:
                    new_tokens[prev_idx] = prev_token.replace("Ws", "Rf")
                elif "RW" in prev_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[prev_idx] = prev_token.replace("RW", "RR")
                elif "WW" in prev_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[prev_idx] = prev_token.replace("WW", "WR")
                else:
                    continue
                # 3. 修改后一项 (Next) - 逻辑反转
                # RW->WR时是 Co->Rfe; 这里 WR->RW 则 Rfe->Co
                next_idx = (i + 1) % n
                next_token = new_tokens[next_idx]

                if "Fr" in next_token:
                    new_tokens[next_idx] = next_token.replace("Fr", "Co")
                elif "RW" in next_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[next_idx] = next_token.replace("RW", "WW")
                elif "RR" in next_token:
                    # 否则，如果以 W 开头，改为 R
                    new_tokens[next_idx] = next_token.replace("RR", "WR")
                else:
                    continue
                print("from : ", cycle_str)
                print("to : ", " ".join(new_tokens))
                mutated_cycles_set.add(" ".join(new_tokens))
                generated_for_this_cycle = True

        if generated_for_this_cycle:
            source_count += 1

    # 保存结果
    try:
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
    # 输入文件
    input_json = "cycles.json"
    # 输出文件
    output_json = "rw_wr_mutated_cycles.json"

    mutate_rw_wr_cycles(input_json, output_json)