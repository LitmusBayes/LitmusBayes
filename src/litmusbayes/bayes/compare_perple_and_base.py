import numpy as np
import pandas as pd

# 请将此处文件名替换为你实际的文件路径
csv_file_path = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat.csv'  # 第一个文件（CSV）
log_file_path = '/home/whq/Desktop/code_list/perple_test/log_C910/log.txt'  # 第二个文件（Log/Text）


def get_stats(ratio_list):
    """计算倍数列表的均值和中位数"""
    if not ratio_list:
        return 0.0, 0.0
    return np.mean(ratio_list), np.median(ratio_list)


def parse_and_compare(csv_path, log_path):
    # --- 1. 读取数据 (保持不变) ---
    try:
        df = pd.read_csv(csv_path)
        csv_data = dict(zip(df['litmus_name'].astype(str), df['num']))
    except Exception as e:
        print(f"读取CSV出错: {e}")
        return

    log_data = {}
    current_test = None
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.endswith(' result:'):
                    current_test = line.replace(' result:', '').strip()
                elif line.startswith('all_allow_litmus:') and current_test:
                    try:
                        vals = line.split(':', 1)[1].split(',')
                        if len(vals) >= 2:
                            log_data[current_test] = float(vals[1].strip())
                    except:
                        continue
    except Exception as e:
        print(f"读取Log出错: {e}")
        return

    # --- 2. 比较倍数 ---
    common_tests = set(csv_data.keys()) & set(log_data.keys())

    # 存储倍数 (Ratio = High / Low)
    csv_ratios = []
    log_ratios = []

    # 记录除以0的情况 (无法计算倍数)
    csv_wins_vs_zero = 0
    log_wins_vs_zero = 0

    equal_count = 0

    print(f"正在分析 {len(common_tests)} 个共同测试用例...\n")

    for test in common_tests:
        s_csv = csv_data[test]
        s_log = log_data[test]

        # CSV 分数更高
        if s_csv > s_log:
            if s_log > 0:
                ratio = s_csv / s_log
                csv_ratios.append(ratio)
            else:
                # 对手是0分，倍数无限大，单独计数
                csv_wins_vs_zero += 1

        # Log 分数更高
        elif s_log > s_csv:
            if s_csv > 0:
                ratio = s_log / s_csv
                log_ratios.append(ratio)
            else:
                # 对手是0分
                log_wins_vs_zero += 1

        else:
            equal_count += 1

    # --- 3. 计算统计 ---
    csv_mean, csv_med = get_stats(csv_ratios)
    print(csv_ratios)
    log_mean, log_med = get_stats(log_ratios)

    # --- 4. 输出倍数统计报告 ---
    print("=" * 50)
    print("           📊 性能倍数对比报告 (Ratio)           ")
    print("=" * 50)
    print(f"注：倍数 = 胜者分数 / 败者分数 (例如 2.0x 代表是对方的2倍)")
    print("-" * 50)

    # CSV 部分
    total_csv_wins = len(csv_ratios) + csv_wins_vs_zero
    print(f"🟢 CSV 胜出 (共 {total_csv_wins} 个):")
    if csv_wins_vs_zero > 0:
        print(f"  Warning: 其中 {csv_wins_vs_zero} 个用例 Log 端为 0 分 (倍数无限大，不计入均值)")

    if len(csv_ratios) > 0:
        print(f"  > 平均倍数 (Mean):   {csv_mean:.2f} x")
        print(f"  > 中位倍数 (Median): {csv_med:.2f} x")
    else:
        print("  (无有效倍数数据)")

    print("-" * 50)

    # Log 部分
    total_log_wins = len(log_ratios) + log_wins_vs_zero
    print(f"🔵 Log 胜出 (共 {total_log_wins} 个):")
    if log_wins_vs_zero > 0:
        print(f"  Warning: 其中 {log_wins_vs_zero} 个用例 CSV 端为 0 分 (倍数无限大，不计入均值)")

    if len(log_ratios) > 0:
        print(f"  > 平均倍数 (Mean):   {log_mean:.2f} x")
        print(f"  > 中位倍数 (Median): {log_med:.2f} x")
    else:
        print("  (无有效倍数数据)")

    print("=" * 50)


if __name__ == '__main__':
    parse_and_compare(csv_file_path, log_file_path)