import json


def normalize_litmus_scores(input_filename, output_filename):
    # 定义作为基准（分母）的 param
    target_param = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    # 读取所有数据
    data = []
    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # 遍历数据，找到每个 litmus 对应的基准分数
    base_scores = {}
    for item in data:
        if item["param"] == target_param:
            base_scores[item["litmus"]] = item["score"]

    # 需要跳过的 litmus 集合（如果 base_score < 0，或者为了防止除零报错而等于 0）
    invalid_litmus = set()
    for litmus, score in base_scores.items():
        if score < 0 or score == 0:  # 附加了 == 0 的检查以防止除以零的错误
            invalid_litmus.add(litmus)

    # 再次遍历数据，进行归一化并写入新文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in data:
            litmus = item["litmus"]

            # 如果该 litmus 没有目标 param 或者是无效的（小于等于0），则跳过整项测试
            if litmus not in base_scores or litmus in invalid_litmus:
                continue

            # 计算归一化后的 score
            base_score = base_scores[litmus]
            if item["score"] < 0:
                continue
            normalized_score = item["score"] / base_score

            # 创建新的字典，避免修改原始数据
            new_item = {
                "litmus": item["litmus"],
                "param": item["param"],
                "score": normalized_score
            }

            # 写入结果文件，每行一个 JSON
            f.write(json.dumps(new_item) + '\n')

    print("归一化完成，结果已保存至", output_filename)


# 使用示例
if __name__ == "__main__":
    input_file = "log_record_init_banana_important_kmeans_sample_2_10.log.validation_cache.jsonl"  # 你的原始数据文件
    output_file = "log_record_init_banana_important_kmeans_sample_2_10_norm.log.validation_cache.jsonl"  # 输出的归一化后的文件

    # 确保自己有建立 input.jsonl 文件之后再运行
    normalize_litmus_scores(input_file, output_file)