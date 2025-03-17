import json
import os


def parse_and_save_positive_scores(log_path, output_path):
    """
    解析日志，提取 score > 0 的 litmus test，并保存为 JSON 文件
    """
    data = {}
    current_test = None
    count = 0

    print(f"正在读取文件: {log_path}")

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 1. 获取当前的 Litmus Test 名称
        if line.endswith(" result:"):
            current_test = line.replace(" result:", "").strip()

        # 2. 解析分数
        elif line.startswith("all_allow_litmus:") and current_test:
            try:
                # 假设格式为 "all_allow_litmus:TestName,Score" 或者类似的 CSV 格式
                # 原代码逻辑：parts[1] 是分数
                content = line.split(":", 1)[1]
                parts = content.split(",")

                if len(parts) >= 2:
                    score = float(parts[1].strip())

                    # --- 关键过滤条件：只存大于 0 的 ---
                    if score > 0:
                        data[current_test] = score
                        count += 1
            except (ValueError, IndexError):
                continue

    # 保存到文件
    print(f"解析完成，找到 {count} 个分值大于 0 的测试用例。")
    print(f"正在写入新文件: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        # indent=4 让文件看起来更美观，ensure_ascii=False 防止中文乱码（如果有）
        json.dump(data, f_out, indent=4, ensure_ascii=False)

    print("写入完成！")


def load_litmus_dict(json_file_path):
    """
    读取生成的 JSON 文件并返回字典
    """
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在 {json_file_path}")
        return {}

    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================
# 主程序入口
# ==========================================
file_path1 = '/home/whq/Desktop/code_list/perple_test/log_C910/log.txt'
# 输出文件路径 (你可以根据需要修改)
new_output_path = './log1_positive_scores.json'

# 1. 执行解析和保存
try:
    parse_and_save_positive_scores(file_path1, new_output_path)

    print("-" * 30)

    # 2. 演示如何读取回字典 (这是你要的读取代码)
    my_litmus_dict = load_litmus_dict(new_output_path)

    # 打印前5个看看效果
    print(f"字典加载成功，共包含 {len(my_litmus_dict)} 个条目。")
    print("前5个数据示例:")
    for i, (k, v) in enumerate(my_litmus_dict.items()):
        if i >= 5: break
        print(f"  {k}: {v}")

except Exception as e:
    print(f"发生错误: {e}")