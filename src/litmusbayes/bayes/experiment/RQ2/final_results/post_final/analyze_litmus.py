import json


def analyze_litmus_data(file_path):
    """
    分析包含 litmus test 数据的 JSON/JSONL 文件。
    统计分数 > 0 的记录中，有多少种不同的 param 组合，及其占比。
    """
    total_positive_scores = 0
    unique_params = set()

    print(f"正在读取文件: {file_path} ...\n")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ 警告：无法解析第 {line_number} 行的 JSON：{line}")
                    continue

                # 判断分数是否大于 0
                score = data.get("score", 0.0)
                if score > 0:
                    total_positive_scores += 1

                    # 获取 param 并转换为 tuple。
                    # 因为 Python 的 list 是不可哈希的，不能放入 set 去重，必须转成 tuple
                    param_list = data.get("param", [])
                    param_tuple = tuple(param_list)
                    unique_params.add(param_tuple)

        # 统计和计算百分比
        if total_positive_scores == 0:
            print("没有找到分数大于 0 的 litmus test 记录。")
            return

        unique_count = len(unique_params)
        percentage = (unique_count / total_positive_scores) * 100

        # 输出结果
        print("-" * 40)
        print("📊 统计结果：")
        print(f"分数 > 0 的 litmus test 总数: {total_positive_scores}")
        print(f"不同的 param 组合种类数: {unique_count}")
        print(f"不同 param 种类占比: {percentage:.2f}%")
        print("-" * 40)

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{file_path}'，请检查文件路径是否正确。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    # 请将 'data.jsonl' 替换为您实际的文件名
    # 如果您的文件就在同一个文件夹，直接写文件名即可
    TARGET_FILE = './median_results.json'

    analyze_litmus_data(TARGET_FILE)