import json
import statistics

# 1. 在这里定义好您的输入文件路径数组
input_files = [
    "perple_log_scores_C910_1.json",
    "perple_log_scores_C910_2.json",
    "perple_log_scores_C910_3.json",
    "perple_log_scores_C910_4.json",
    "perple_log_scores_C910_5.json",
    "perple_log_scores_C910_6.json",
    "perple_log_scores_C910_7.json",
    "perple_log_scores_C910_8.json",

]

# 2. 定义输出文件路径
output_file = "median_results.json"


def process_litmus_medians(files, out_file):
    # 用于按 litmus 分组聚合数据：{ "litmus_name": [score1, score2, ...] }
    litmus_data = {}

    # 读取并收集所有分数
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    # 尝试将整个文件作为一个标准的 JSON 字典读取
                    data = json.load(f)
                    for litmus, score in data.items():
                        if litmus not in litmus_data:
                            litmus_data[litmus] = []
                        litmus_data[litmus].append(score)

                except json.JSONDecodeError:
                    # 如果不是标准的 JSON 字典（例如没有外层的 {}，或者是多行键值对）
                    # 回退到逐行解析的方式
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        # 跳过空行或者只包含大括号的行
                        if not line or line in ("{", "}"):
                            continue

                        # 去掉行末可能存在的逗号，以便单独解析
                        if line.endswith(","):
                            line = line[:-1]

                        try:
                            # 补齐大括号当做单独的 JSON 对象解析: {"key": value}
                            parsed_line = json.loads("{" + line + "}")
                            for litmus, score in parsed_line.items():
                                if litmus not in litmus_data:
                                    litmus_data[litmus] = []
                                litmus_data[litmus].append(score)
                        except json.JSONDecodeError:
                            print(f"警告：跳过无法解析的行 -> {line}")

        except FileNotFoundError:
            print(f"警告：找不到文件 -> {file_path}")

    # 计算中位数
    output_results = {}
    processed_count = 0

    for litmus, scores in litmus_data.items():
        if not scores:
            continue

        # 维持您原有的中位数计算逻辑
        median_score = statistics.median(scores)
        if sum(scores) == -8:
            median_score = -1.0
        else:
            new_scores = []
            for score in scores:
                new_scores.append(max(score, 0.0))
            median_score = statistics.median(new_scores)

        # 打印符合特定条件的调试信息
        if sum(scores) > 0 and median_score == 0.0:
            print(f"[警告 Sum>0 但 Median=0.0] {litmus}")
            print(f"Scores: {scores}")

        zero_count = sum(1 for s in scores if s == 0.0)
        if zero_count == 4:
            print(f"[警告 正好 4 个 0] {litmus}")

        # 记录中位数结果
        output_results[litmus] = median_score
        processed_count += 1

    # 将结果写入新文件（保存为标准的 JSON 字典格式）
    with open(out_file, 'w', encoding='utf-8') as f_out:
        # indent=4 会让输出像您提供的那样带有缩进和换行
        json.dump(output_results, f_out, indent=4, ensure_ascii=False)

    print(f"\n处理完成！共聚合了 {processed_count} 个 litmus 测试的中位数。")
    print(f"结果已保存至: {out_file}")


if __name__ == "__main__":
    process_litmus_medians(input_files, output_file)