import json


def extract_scores_by_param(input_file, output_file):
    # 1. 定义我们要寻找的目标参数
    target_param = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

    # 用于存储结果的字典
    result_map = {}

    print(f"正在读取文件: {input_file} ...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    current_param = data.get("param")

                    # 2. 比较参数是否与目标一致
                    # Python 中列表可以直接进行相等性比较
                    if current_param == target_param:
                        litmus_name = data.get("litmus")
                        score = data.get("score")

                        # 存入字典: "litmus_name": score
                        result_map[litmus_name] = score

                except json.JSONDecodeError:
                    print("警告: 发现无法解析的行，已跳过。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    # 3. 将结果写入新文件
    print(f"找到 {len(result_map)} 个匹配项，正在写入 {output_file} ...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # indent=4 会让输出格式化，变成你想要的竖排 kv 结构
        json.dump(result_map, f, indent=4, ensure_ascii=False)

    print("完成！")


# --- 配置区域 ---

# 输入文件名 (你的原始数据)
input_filename = 'log/cache_0.jsonl'

# 输出文件名 (生成的新文件)
output_filename = 'log/baseline_scores_0.json'

# --- 执行 ---
if __name__ == "__main__":
    extract_scores_by_param(input_filename, output_filename)