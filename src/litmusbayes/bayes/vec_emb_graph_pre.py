import json
import os


def load_valid_names_from_graph_file(graph_file_path):
    """
    从图数据文件(文件2)中提取所有合法的 'name' 字段。
    支持标准 JSON 列表 [...] 或 堆叠的 JSON [...] [...]
    """
    valid_names = set()

    # 1. 读取文件内容
    with open(graph_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 解析 JSON
    # 使用流式解析以应对可能的格式 (列表 或 堆叠列表)
    decoder = json.JSONDecoder()
    pos = 0

    while pos < len(content):
        # 跳过空白
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break

        try:
            obj, end_pos = decoder.raw_decode(content[pos:])

            # 提取名字
            if isinstance(obj, list):
                for item in obj:
                    if 'name' in item:
                        valid_names.add(item['name'])
            elif isinstance(obj, dict):
                if 'name' in obj:
                    valid_names.add(obj['name'])

            pos += end_pos
        except json.JSONDecodeError:
            print(f"⚠️ 解析警告: 在位置 {pos} 遇到非 JSON 内容，停止解析剩余部分。")
            break

    print(f"✅ 白名单加载完毕，共找到 {len(valid_names)} 个唯一的 Litmus Test 名字。")
    return valid_names


def filter_performance_log(log_file_path, valid_names, output_path):
    """
    根据 valid_names 过滤 log 文件
    """
    kept_count = 0
    dropped_count = 0

    with open(log_file_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                litmus_name = entry.get('litmus')

                # --- 核心判断逻辑 ---
                if litmus_name in valid_names:
                    # 在白名单里，保留
                    f_out.write(line + '\n')
                    kept_count += 1
                else:
                    # 不在白名单里，丢弃
                    dropped_count += 1

            except json.JSONDecodeError:
                print(f"⚠️ 跳过损坏的行: {line[:50]}...")
                continue

    print("-" * 30)
    print(f"处理完成！")
    print(f"原文件: {log_file_path}")
    print(f"保留条目: {kept_count}")
    print(f"过滤条目: {dropped_count}")
    print(f"结果已保存至: {output_path}")


# ================= 配置与运行 =================

# 1. 你的性能日志文件 (文件1, 格式 {"litmus": ..., "score": ...})
perf_log_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no.jsonl"

# 2. 你的图数据文件 (文件2, 格式 [{"name": ..., "edges": ...}])
# 这里的路径应该是你之前过滤生成的那个 filtered_result.json 或者原始的 graph json
graph_data_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"

# 3. 输出文件路径
output_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_for_graph.jsonl"

if __name__ == "__main__":
    if os.path.exists(perf_log_file) and os.path.exists(graph_data_file):
        # 第一步：建立白名单
        valid_name_set = load_valid_names_from_graph_file(graph_data_file)
        print(len(valid_name_set))
        # 第二步：过滤日志
        filter_performance_log(perf_log_file, valid_name_set, output_file)
    else:
        print("❌ 文件不存在，请检查路径。")