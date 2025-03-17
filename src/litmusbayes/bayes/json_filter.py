import json


def process_litmus_data(base_file, target_files, output_file):
    """
    处理Litmus测试数据的JSON文件。
    """

    # 1. 读取基准文件，构建合法 litmus 名称的集合 (Set)
    # 使用 Set 可以实现 O(1) 的快速查找
    allowed_litmus_names = set()

    print(f"正在读取基准文件: {base_file} ...")
    try:
        with open(base_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    allowed_litmus_names.add(data.get("litmus"))
                except json.JSONDecodeError:
                    print(f"警告: {base_file} 中有一行无法解析，已跳过。")
    except FileNotFoundError:
        print(f"错误: 找不到基准文件 {base_file}")
        return

    print(f"基准文件读取完毕，共找到 {len(allowed_litmus_names)} 个唯一的 litmus 测试名。")

    # 2. 处理目标文件并聚合数据
    # 使用字典来存储数据: key = (litmus_name, tuple(param)), value = [score1, score2, ...]
    # 注意：param 是列表，不可哈希，必须转为元组 (tuple) 才能作为字典的 key
    aggregation_map = {}

    for file_path in target_files:
        print(f"正在处理文件: {file_path} ...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue

                    try:
                        data = json.loads(line)
                        name = data.get("litmus")
                        score = data.get("score")
                        param = data.get("param")

                        # 过滤条件 A: litmus name 必须在基准文件中
                        if name not in allowed_litmus_names:
                            continue

                        # 过滤条件 B: 排除 score 为 -1 的项
                        # 注意浮点数比较，但 -1 通常是精确整数，这里直接比较即可
                        if score == -1:
                            continue

                        # 生成唯一键 (name, param_tuple)
                        key = (name, tuple(param))

                        if key not in aggregation_map:
                            aggregation_map[key] = []

                        aggregation_map[key].append(score)

                    except json.JSONDecodeError:
                        print(f"警告: {file_path} 中有一行无法解析，已跳过。")
        except FileNotFoundError:
            print(f"警告: 找不到文件 {file_path}，已跳过。")

    # 3. 计算平均值并写入结果
    print(f"正在计算平均值并写入 {output_file} ...")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        count = 0
        for (name, param_tuple), scores in aggregation_map.items():
            if not scores:
                continue

            # 计算平均分
            avg_score = sum(scores) / len(scores)

            # 构建输出对象
            result_obj = {
                "litmus": name,
                "param": list(param_tuple),  # 转回列表
                "score": avg_score
            }

            # 写入文件（每行一个 JSON）
            out_f.write(json.dumps(result_obj) + "\n")
            count += 1

    print(f"处理完成！成功生成 {count} 条记录到 {output_file}")


# --- 配置区域 ---

# 你的第一个文件（基准文件）
base_json_file = 'log_record_bayes.log.cache_sum_70_no_norm_gt_0_for_graph.jsonl'

# 你的其他六个文件 (请在此处修改为实际文件名)
other_json_files = [
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.28.log.cache.jsonl',
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.46.log.cache.jsonl',
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.48.log.cache.jsonl',
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.58.log.cache.jsonl',
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.61.log.cache.jsonl',
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.112.log.cache.jsonl',
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.139.log.cache.jsonl',
    '/home/whq/Desktop/code_list/perple_test/bayes_stat/log/log_record_10.42.0.238.log.cache.jsonl',
]

# 输出文件名
output_json_file = 'log/cache_0.jsonl'

# --- 执行脚本 ---
if __name__ == "__main__":
    # 为了演示，如果你没有实际文件，代码会报错。
    # 请确保上述 filenames 对应你本地的真实文件。
    process_litmus_data(base_json_file, other_json_files, output_json_file)