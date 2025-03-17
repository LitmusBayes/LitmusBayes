import json
import os


def load_names_from_json(json_path):
    """
    从文件二中提取所有的 'name' 字段。
    兼容 JSON Array ([...])、JSON Lines 和拼接的 JSON 对象。
    """
    valid_names = set()

    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return valid_names

    print(f"正在从 {json_path} 加载允许的名称...")

    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # 跳过空白字符
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break

        try:
            # 尝试解析下一个 JSON 对象
            obj, end_pos = decoder.raw_decode(content[pos:])
            pos += end_pos

            # 处理解析出的对象
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and 'name' in item:
                        valid_names.add(item['name'])
            elif isinstance(obj, dict) and 'name' in obj:
                valid_names.add(obj['name'])

        except json.JSONDecodeError:
            # 如果解析失败，尝试跳过该位置（简单的容错处理）
            pos += 1

    print(f"--> 从文件二中找到了 {len(valid_names)} 个唯一的名称。")
    return valid_names


def filter_text_file(txt_path, output_path, valid_names):
    """
    过滤文件一，仅保留 name 在 valid_names 中的行。
    """
    if not os.path.exists(txt_path):
        print(f"错误: 找不到文件 {txt_path}")
        return

    print(f"正在处理 {txt_path} ...")

    kept_count = 0
    total_count = 0

    with open(txt_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            total_count += 1

            # 文件一格式: "Name: Body..."
            # 我们以第一个冒号分割
            if ":" in line:
                name_part = line.split(":", 1)[0].strip()

                # 检查该名称是否在文件二的集合中
                if name_part in valid_names:
                    f_out.write(line + "\n")
                    kept_count += 1
            else:
                # 如果行内没有冒号，根据需要决定是否保留，这里默认不保留
                pass

    print(f"--> 处理完成。")
    print(f"    原始行数: {total_count}")
    print(f"    保留行数: {kept_count}")
    print(f"    结果已保存至: {output_path}")



graph_json_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"
litmus_cycle_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3.log"  # 存放原始文本的文件

# ================= 配置区域 =================
# 请在这里替换你的实际文件名
file1_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3.log"  # 你的第一个文件（待过滤的文本）
file2_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"  # 你的第二个文件（提供 name 列表的 JSON）
output_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3_aligned.log"  # 输出结果文件
# ===========================================

if __name__ == "__main__":
    # 1. 获取白名单
    allow_list = load_names_from_json(file2_path)

    # 2. 只有当白名单不为空时才进行过滤
    if allow_list:
        filter_text_file(file1_path, output_file_path, allow_list)
    else:
        print("警告: 未能从文件二中提取到任何名称，未执行过滤。")