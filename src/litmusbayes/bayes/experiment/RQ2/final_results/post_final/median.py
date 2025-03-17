import json
import statistics

# 1. 在这里定义好您的输入文件路径数组
input_files = [
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_8.log.validation_cache.jsonl",
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_7.log.validation_cache.jsonl",
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_6.log.validation_cache.jsonl",
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_5.log.validation_cache.jsonl",
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_4.log.validation_cache.jsonl",
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_3.log.validation_cache.jsonl",
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_2.log.validation_cache.jsonl",
    "log_record_best_post_final_kmeans_cross_10_factor_2_banana_1.log.validation_cache.jsonl",
]
# 2. 定义输出文件路径
output_file = "median_results.json"


def process_litmus_medians(files, out_file):
    # 用于按 litmus 分组聚合数据：{ "litmus_name": {"param": [...], "scores": []} }
    litmus_data = {}

    # 读取并收集所有分数
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        litmus = data.get("litmus")
                        if not litmus:
                            continue

                        param = data.get("param", [])
                        score = data.get("score", 0.0)

                        # 规则：如果 score 是 -1，视为 0
                        if score == -1 or score == -1.0:
                            score = 0.0

                        # 初始化或追加分数
                        if litmus not in litmus_data:
                            litmus_data[litmus] = {
                                "param": param,
                                "scores": []
                            }
                        litmus_data[litmus]["scores"].append(score)

                    except json.JSONDecodeError:
                        print(f"警告：跳过无法解析的 JSON 行 -> {line}")

        except FileNotFoundError:
            print(f"警告：找不到文件 -> {file_path}")

    # 计算中位数并按照原格式写入新文件
    processed_count = 0
    with open(out_file, 'w', encoding='utf-8') as f_out:
        for litmus, info in litmus_data.items():
            scores = info["scores"]
            if not scores:
                continue

            # 计算中位数
            median_score = statistics.median(scores)
            if sum(scores) == -8:
                median_score = -1
            else:
                new_scores = []
                for score in scores:
                    new_scores.append(max(score,0.0))
                median_score = statistics.median(new_scores)
            if sum(scores) > 0 and median_score == 0.0:
                print(litmus)
                print(scores)
            # 组装原有格式的 JSON
            output_record = {
                "litmus": litmus,
                "param": info["param"],
                "score": median_score
            }

            # 写入新文件，每行一个 JSON 对象
            f_out.write(json.dumps(output_record) + "\n")
            processed_count += 1

    print(f"处理完成！共聚合了 {processed_count} 个 litmus 测试的中位数。")
    print(f"结果已保存至: {out_file}")


if __name__ == "__main__":
    process_litmus_medians(input_files, output_file)