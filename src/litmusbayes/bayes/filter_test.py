import json
import sys
from collections import defaultdict


def report_duplicates_aggregated(file_path):
    # 使用 defaultdict 方便存储列表
    # Key: (litmus, param_tuple)
    # Value: list of { 'line_num': int, 'content': string }
    grouped_entries = defaultdict(list)

    print(f"正在读取并分析文件: {file_path} ...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    litmus_val = data.get("litmus")
                    param_val = data.get("param")

                    # 同样需要转 tuple 才能作为字典 key
                    param_tuple = tuple(param_val) if isinstance(param_val, list) else param_val

                    # 组合 Key
                    key = (litmus_val, param_tuple)

                    # 将当前行的信息存入对应的组中
                    grouped_entries[key].append({
                        "line_num": line_num,
                        "content": line
                    })

                except json.JSONDecodeError:
                    print(f"[警告] 第 {line_num} 行 JSON 格式错误，跳过。")
                except Exception as e:
                    print(f"[警告] 第 {line_num} 行处理出错: {e}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        return

    # --- 分析完成，开始生成报告 ---

    print("-" * 80)
    print("重复条目报告")
    print("-" * 80)

    duplicate_groups_found = 0

    # 遍历所有组，只处理长度 > 1 的（即有重复的）
    for (litmus, param), entries in grouped_entries.items():
        if len(entries) > 1:
            duplicate_groups_found += 1
            print(f"【重复组 #{duplicate_groups_found}】")
            print(f"Key信息: Litmus='{litmus}', Param={list(param)}")
            print(f"重复数量: {len(entries)} 条")
            print("详细内容:")

            for entry in entries:
                print(f"  [行 {entry['line_num']}]: {entry['content']}")

            print("-" * 40)  # 分隔线

    if duplicate_groups_found == 0:
        print("未发现任何重复的 litmus + param 组合。")
    else:
        print(f"报告结束。共发现 {duplicate_groups_found} 组重复数据。")


if __name__ == "__main__":
    file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache2.jsonl"
    report_duplicates_aggregated(file_path)