import sys


def load_data(filepath):
    """
    读取文件，将每行解析为 key-value 字典。
    假设每行格式为: "SB+po-ctrlfencei+pos-po-addrs 0.057049"
    """
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    # 尝试将第二部分转换为浮点数
                    val = float(parts[1])
                    data[key] = val
                except ValueError:
                    print(f"警告: 文件 {filepath} 第 {line_num} 行的数值无法解析为浮点数: {line}")
            else:
                print(f"警告: 文件 {filepath} 第 {line_num} 行格式不符: {line}")
    return data


def compare_files(file1, file2):
    print(f"正在比较:\n文件1: {file1}\n文件2: {file2}\n")

    data1 = load_data(file1)
    data2 = load_data(file2)

    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    only_in_file1 = keys1 - keys2
    only_in_file2 = keys2 - keys1
    common_keys = keys1 & keys2

    # 1. 打印独有的项
    if only_in_file1:
        print(f"=== 只在 文件1 中存在的项 ({len(only_in_file1)} 个) ===")
        for k in only_in_file1:
            print(f"  {k}: {data1[k]}")
        print()

    if only_in_file2:
        print(f"=== 只在 文件2 中存在的项 ({len(only_in_file2)} 个) ===")
        for k in only_in_file2:
            print(f"  {k}: {data2[k]}")
        print()

    # 2. 比较共有项的数值差异
    print(f"=== 共有项的数值比较 ({len(common_keys)} 个) ===")
    diff_records = []
    for k in common_keys:
        val1 = data1[k]
        val2 = data2[k]
        diff = val2 - val1
        # 只记录有差异的项 (可以设置一个容差，比如 abs(diff) > 1e-9)
        if diff != 0:
            diff_records.append((k, val1, val2, diff))

    if not diff_records:
        print("所有共有项的数值完全一致！")
    else:
        # 按照差异大小排序（绝对值降序）
        diff_records.sort(key=lambda x: abs(x[3]), reverse=True)
        print(f"{'测试项名称':<45} | {'文件1数值':<20} | {'文件2数值':<20} | {'差值 (文件2 - 文件1)'}")
        print("-" * 110)
        for record in diff_records:
            k, v1, v2, diff = record
            print(f"{k:<45} | {v1:<20.10e} | {v2:<20.10e} | {diff:+.10e}")


if __name__ == "__main__":


    file1_path = "./log_1.txt"
    file2_path = "./log_2.txt"

    compare_files(file1_path, file2_path)