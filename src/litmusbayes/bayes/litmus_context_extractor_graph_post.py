import json
import ast
import os
import re


def load_target_states(filepath):
    """
    解析第二个文件，建立一个 {name: state_dict} 的映射表。
    """
    target_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # 分割文件名和对应的状态
                if ': ' in line:
                    name_part, state_str = line.split(': ', 1)
                else:
                    name_part, state_str = line.split(':', 1)

                clean_name = name_part.replace('.litmus', '').strip()

                try:
                    state_dict = json.loads(state_str)
                except json.JSONDecodeError:
                    state_dict = ast.literal_eval(state_str)

                target_map[clean_name] = state_dict

            except Exception as e:
                print(f"Warning: 解析行出错 '{line}': {e}")
                continue

    return target_map


def load_stacked_json_arrays(filename):
    """
    流式读取堆叠的 JSON 数组
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    all_data = []

    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end_pos = decoder.raw_decode(content[pos:])
            if isinstance(obj, list):
                all_data.extend(obj)
            else:
                all_data.append(obj)
            pos += end_pos
        except json.JSONDecodeError as e:
            print(f"❌ 解析错误发生在字符位置 {pos}: {e}")
            break
    return all_data


def save_compact_json(data, output_path):
    """
    保存 JSON，但将纯数字的列表压缩到一行显示。
    例如：
    [
      1,
      0
    ]
    会被压缩为 [1, 0]
    """
    # 1. 先生成标准的带缩进 JSON 字符串
    json_str = json.dumps(data, indent=2)

    # 2. 定义正则：匹配 [ 数字, 空格, 换行, 逗号 ] 组成的块
    # 解释：\[\s* 匹配开头括号和空白
    # ([\d\s,-]+) 捕获组：匹配数字、空白、逗号、负号
    # \s*\] 匹配结尾空白和括号
    pattern = r'\[\s*([\d\s,-]+?)\s*\]'

    def compact_match(match):
        # 获取方括号内的内容
        content = match.group(1)
        # 如果内容里没有换行符，说明原本就是一行的，不用处理（或者很短）
        if '\n' not in content:
            return match.group(0)

        # 去掉换行和多余空格，重新拼接
        # 将 "1,\n  0" 变成 "1, 0"
        compact_content = ", ".join([x.strip() for x in content.split(',') if x.strip()])
        return f"[{compact_content}]"

    # 3. 执行正则替换
    compact_json_str = re.sub(pattern, compact_match, json_str)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(compact_json_str)


def filter_data_json(json_path, target_states, output_path):
    source_data = load_stacked_json_arrays(json_path)
    filtered_list = []

    # 用于记录哪些名字成功匹配上了
    matched_names_set = set()

    for item in source_data:
        name = item.get('name')
        raw_state = item.get('state')

        # 1. 检查名字是否在目标列表中
        if name not in target_states:
            continue

        # 2. 解析 JSON 中的 state
        try:
            current_state_dict = ast.literal_eval(raw_state)
        except:
            try:
                current_state_dict = json.loads(raw_state)
            except:
                print(f"Error parsing state for {name}")
                continue

        # 3. 比较状态
        expected_state = target_states[name]

        if current_state_dict == expected_state:
            filtered_list.append(item)
            matched_names_set.add(name)  # 记录匹配成功的名字

    # === 保存结果 (使用紧凑格式) ===
    save_compact_json(filtered_list, output_path)

    # === 计算未匹配的项 ===
    # 目标文件中所有的名字 - 成功匹配并保存的名字 = 未匹配的名字
    all_target_names = set(target_states.keys())
    unmatched_names = all_target_names - matched_names_set

    print("-" * 30)
    print(f"处理完成。")
    print(f"源文件条目数: {len(source_data)}")
    print(f"保留条目数: {len(filtered_list)}")
    print(f"结果已保存至: {output_path}")
    print("-" * 30)

    if unmatched_names:
        print(f"⚠️  文件2中有 {len(unmatched_names)} 个 Litmus Test 在文件1中未找到匹配项 (或状态不符):")
        # 排序后打印前20个，避免刷屏，如果需要全部可以去掉切片
        sorted_unmatched = sorted(list(unmatched_names))
        for missing in sorted_unmatched:
            print(f"  - {missing}")
    else:
        print("✅ 文件2中的所有 Litmus Test 都成功匹配到了。")


# ================= 使用示例 =================

# 1. 定义文件名
source_file = "/src/slide/bayes/make_new_litmus/litmus_vector_new_graph.jsonl"
target_file = "/src/slide/bayes/make_new_litmus/result.log"
output_file = "/src/slide/bayes/make_new_litmus/litmus_vector_new_graph_filter.jsonl"

# 2. 运行
if os.path.exists(source_file) and os.path.exists(target_file):
    target_map = load_target_states(target_file)
    filter_data_json(source_file, target_map, output_file)
else:
    print(f"文件不存在，请检查路径:\n{source_file}\n{target_file}")