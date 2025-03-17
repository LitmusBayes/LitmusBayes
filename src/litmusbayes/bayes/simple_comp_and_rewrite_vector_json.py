import json


def filter_json_by_blacklist(blacklist_path, json_path, output_path):
    # 1. 解析文件 1，提取黑名单中的 litmus name
    blacklist = set()
    with open(blacklist_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                # 提取 "|" 之前的内容并去除首尾空格
                name = line.split('|')[0].strip()
                if name:
                    blacklist.add(name)

    print(f"成功加载黑名单，共 {len(blacklist)} 个项。")

    # 2. 读取并处理 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("错误：JSON 文件格式不正确，请确保它是一个完整的 JSON 数组。")
            return

    # 3. 执行过滤逻辑
    # 如果 data 是列表（通常是这种情况）
    if isinstance(data, list):
        filtered_data = [item for item in data if item.get("name") not in blacklist]
        removed_count = len(data) - len(filtered_data)
    else:
        # 如果 JSON 根对象是字典（单个对象）
        if data.get("name") in blacklist:
            filtered_data = {}
            removed_count = 1
        else:
            filtered_data = data
            removed_count = 0

    # 4. 将结果保存到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"处理完成！")
    print(f"过滤掉的项数: {removed_count}")
    print(f"剩余项数: {len(filtered_data) if isinstance(filtered_data, list) else 1}")
    print(f"结果已保存至: {output_path}")


# --- 配置路径 ---
blacklist_file = '/home/whq/Desktop/code_list/perple_test/bayes_stat/filtered_litmus_log.txt'  # 你的第一个文件
input_json = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"  # 原始 JSON 文件
output_json = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2_gt0.jsonl"  # 过滤后的 JSON 文件

filter_json_by_blacklist(blacklist_file, input_json, output_json)