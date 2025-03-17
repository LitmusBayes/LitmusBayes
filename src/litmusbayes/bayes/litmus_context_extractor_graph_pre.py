import os
import re
import json


def generate_litmus_log(root_dir, log_file_path):
    """
    遍历目录，提取 .litmus 文件中 exists 后的条件，
    并按行写入 .log 文件。
    特殊处理：纯变量名（如 x, y）自动加上 []，寄存器名（如 0:x5）保持原样。
    """
    # 1. 匹配 exists(...) 括号内的内容
    pattern_exists = re.compile(r"exists\s*\(([^)]+)\)", re.IGNORECASE)

    # 2. 分割逻辑：匹配 /\ 以及周围可能存在的空格
    pattern_split = re.compile(r'\s*/\\\s*')

    print(f"正在扫描: {root_dir}")
    print(f"输出目标: {log_file_path}")

    count = 0

    with open(log_file_path, 'w', encoding='utf-8') as log_f:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".litmus"):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                            match = pattern_exists.search(content)
                            if match:
                                condition_str = match.group(1).strip()

                                # 使用正则分割，处理可能的空格问题
                                conditions = pattern_split.split(condition_str)

                                result_dict = {}
                                for cond in conditions:
                                    if '=' in cond:
                                        key, value = cond.split('=', 1)
                                        key = key.strip()
                                        value = value.strip()

                                        # --- 核心修改逻辑开始 ---
                                        # 如果 key 中不包含冒号(说明不是 0:x5 这种寄存器)
                                        # 并且 key 也没有被括号包围
                                        if ':' not in key and not (key.startswith('[') and key.endswith(']')):
                                            key = f"[{key}]"
                                        # --- 核心修改逻辑结束 ---

                                        # 数字转 Int
                                        if value.isdigit():
                                            result_dict[key] = int(value)
                                        else:
                                            result_dict[key] = value

                                # 生成 JSON
                                json_str = json.dumps(result_dict, ensure_ascii=False)

                                # 写入 log
                                log_f.write(f"{file}: {json_str}\n")
                                count += 1

                    except Exception as e:
                        print(f"读取出错 {file}: {e}")

    print(f"完成！已生成日志文件，共 {count} 条记录。")


if __name__ == "__main__":
    # 路径保持不变
    folder = "/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_output"
    output_log = "/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_output/result.log"

    generate_litmus_log(folder, output_log)