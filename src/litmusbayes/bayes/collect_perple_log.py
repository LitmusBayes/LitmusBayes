import os
import re
import json


def process_logs(folder_path, output_json):
    results = {}

    # [修改点 1]：确保匹配的是真实数字（支持整数和浮点数），避免匹配到单独的 '.'
    heuristic_pattern = re.compile(r'heuristic statistic:\s*(\d+)')
    time_pattern = re.compile(r'Time.*?(\d+\.\d+|\d+)')

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if not os.path.isfile(filepath):
            continue

        name = filename.split('_')[0]
        name = name.replace('.log', '')

        total_heuristic = 0
        time_value = None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    h_match = heuristic_pattern.search(line)
                    if h_match:
                        total_heuristic += int(h_match.group(1))
                        continue

                    if 'Time' in line:
                        t_match = time_pattern.search(line)
                        if t_match:
                            try:
                                time_value = float(t_match.group(1))
                            except ValueError:
                                # [修改点 2]：万一还是转换失败，跳过该行而不是直接终止读取该文件
                                continue

        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")
            continue

        if time_value is not None and time_value > 0:
            score = total_heuristic / time_value
            results[name] = score
        else:
            print(f"警告: 文件 {filename} 未找到有效时间数值，跳过。")

    try:
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)
        print(f"\n处理完成！成功解析了 {len(results)} 个文件，结果已保存至 {output_json}")
    except Exception as e:
        print(f"写入 JSON 文件时出错: {e}")
# ==========================================
# 使用配置
# ==========================================
if __name__ == "__main__":
    # 替换为你存放日志文件的文件夹路径，例如 './logs'
    TARGET_FOLDER = '/home/whq/Desktop/code_list/perple_test/perple_log_banana'
    # 期望生成的 JSON 文件名
    OUTPUT_JSON_FILE = 'perple_log_scores_banana.json'

    # 如果目标文件夹存在，则执行脚本
    if os.path.exists(TARGET_FOLDER):
        process_logs(TARGET_FOLDER, OUTPUT_JSON_FILE)
    else:
        print(f"错误: 找不到文件夹 '{TARGET_FOLDER}'，请检查路径是否正确。")
