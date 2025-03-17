
import json
import re
import statistics


def parse_log1(content):
    """
    解析日志1 (Text格式)
    逻辑: 找到测试名称行，然后找到该测试下的 all_allow_litmus 行提取分数
    """
    data = {}
    current_test = None

    # 按行处理
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()

        # 1. 识别测试名称行 (以 result: 结尾)
        if line.endswith(" result:"):
            # 移除 " result:" 获取测试名
            current_test = line.replace(" result:", "").strip()

        # 2. 识别目标分数行 (all_allow_litmus)
        elif line.startswith("all_allow_litmus:") and current_test:
            # 格式: all_allow_litmus: count, score
            # 分割字符串提取数字
            try:
                parts = line.split(":")[1].split(",")
                if len(parts) >= 2:
                    score = float(parts[1].strip())
                    data[current_test] = score
            except (ValueError, IndexError):
                pass

    return data

def parse_log2(content):
    """
    解析日志2 (JSON Lines格式)
    逻辑: 读取每一行JSON，对于相同的 litmus，只保留最大的 score
    """
    data = {}

    lines = content.strip().split('\n')
    for line in lines:
        if not line.strip(): continue
        try:
            entry = json.loads(line)
            name = entry.get("litmus")
            print(entry)
            if entry.get("score", 0) is None:
                continue
            score = float(entry.get("score", 0))

            if name:
                # 如果该测试已存在，取当前分数与已知最大值的较大者
                if name in data:
                    data[name] = max(data[name], score)
                else:
                    data[name] = score
        except json.JSONDecodeError:
            continue

    return data

def compare_logs(log1_data, log2_data):
    """
    比较两个数据集
    返回: 计数, 详细列表
    """
    count = 0
    details = []
    summary = []
    zero1 = 0
    zero2 = 0
    all_zero = 0
    not_zero = 0
    not_pass = []
    times1 = 0
    times2 = 0
    exp = 0
    # 遍历 Log2 中所有的测试 (因为我们要找的是 Log2 > Log1)
    for test_name, score2 in log2_data.items():
        # 只有当 Log1 中也有这个测试时才比较
        if test_name in log1_data:
            score1 = log1_data[test_name]

            # 严格大于
            if score1 == score2 and score1 == 0:
                all_zero += 1
            elif score2 > score1:
                count += 1
                details.append({
                    "test": test_name,
                    "log1_score": score1,
                    "log2_max_score": score2
                })
                if score1 != 0:
                    summary.append(score2/score1)
                    times1 = times1 + 3 / score1
                    times2 = times2 + 3 / score2
                    not_zero += 1
                else:
                    zero1 += 1
            elif score1 > score2 and score2 != 0 and score2 != -1:
                not_pass.append((test_name, score1, score2))
                summary.append(score2 / score1)
                times2 = times2 + 3 / score2
                times1 = times1 + 3 / score1
                not_zero += 1
            elif score2 == score1 and score2 != 0 and score2 != -1:
                summary.append(score2 / score1)
                times2 = times2 + 3 / score2
                times1 = times1 + 3 / score1
                not_zero += 1
            elif score2 == -1 or score2 == 0:
                if score2 == -1:
                    exp += 1
                not_zero += 1
                zero2 += 1
                print("zero2", test_name)
    if summary:

        mean_val = sum(summary) / len(summary)
        median_val = statistics.median(summary)  # 计算中位数
    else:
        mean_val = 0
        median_val = 0
    return count, details, mean_val, median_val, zero1, zero2, all_zero, not_zero, not_pass, times1, times2, exp

# ==========================================
# 这里填入你的数据进行测试
# ==========================================


# 读取文件内容
with open('/home/whq/Desktop/code_list/perple_test/log_C910/log.txt', 'r') as f:
    log1_raw = f.read()

# with open('/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no.jsonl', 'r') as f:
with open('/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_for_best10.jsonl',
              'r') as f:
    log2_raw = f.read()

# 剩下的代码调用 parse_log1, parse_log2 和 compare_logs 保持不变
# 执行解析
data1 = parse_log1(log1_raw)
data2 = parse_log2(log2_raw)

# 执行比较
result_count, result_details, mean_val, median_val, zero1, zero2, all_zero, not_zero, not_pass, times1, times2, exp = compare_logs(data1, data2)

print(f"--- 解析结果 ---")
print(f"Log1 数据: {json.dumps(data1, indent=2)}")
print(f"Log2 数据 (取最大值后): {json.dumps(data2, indent=2)}")
print(f"\n--- 比较结果 ---")
print(f"Log2 严格大于 Log1 的测试数量: {result_count}")

if result_count > 0:
    print("详细列表:")
    for item in result_details:
        print(f"Test: {item['test']} | Log1: {item['log1_score']} < Log2: {item['log2_max_score']}")
else:
    print("(基于你提供的片段，没有发现 Log2 分数大于 Log1 的情况)")

print(f"not pass:{len(not_pass)}")
for litmus_name, score1, score2 in not_pass:
    print(f"litmus_name:{litmus_name}, score1:{score1}, score2:{score2}")

print(f"result_count: {result_count}")
print(f"mean_val: {mean_val}")
print(f"median_val: {median_val}")
print(f"zero1: {zero1}")
print(f"zero2: {zero2}")
print(f"not_zero: {not_zero}")
print(f"all_zero: {all_zero}")
print(f"times1:{times1}")
print(f"times2:{times2}")
print(f"times1/times2:{times1/times2}")
print(f"exp:{exp}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 1. 数据筛选与计算
# ==========================================
common_tests = set(data1.keys()) & set(data2.keys())
ratios = []

for test in common_tests:
    s1 = data1[test]
    s2 = data2[test]

    if s1 > 0:
        r = s2 / s1
        if r >= 1.0:  # 只看非退步的
            ratios.append(r)

# ==========================================
# 2. 定义区间 (标签改为全英文，避开字体问题)
# ==========================================
bins = [1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 5.0, 10.0, 10000]

# 使用英文标签，避免 Linux 字体报错
labels = [
    '1.0 - 1.05x\n',  # 微幅
    '1.05 - 1.1x\n',  # 小幅
    '1.1 - 1.2x\n',  # 稳健
    '1.2 - 1.5x\n',  # 显著
    '1.5 - 2.0x\n',  # 高效
    '2.0 - 5.0x\n',  # 飞跃
    '5.0 - 10x\n',  # 质变
    '> 10x\n(Extreme)'  # 极速
]

df_ratios = pd.DataFrame({'ratio': ratios})
df_ratios['category'] = pd.cut(df_ratios['ratio'], bins=bins, labels=labels, right=False)
counts = df_ratios['category'].value_counts().sort_index()

# ==========================================
# 3. 绘图设置 (修复字体报错)
# ==========================================
# plt.style.use('dark_background')

# 【关键修改】指定 Linux 必有的通用字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
# 预览背景色 (保存时会透明)
fig.patch.set_facecolor('#001f3f')
ax.set_facecolor('#001f3f')

# ==========================================
# 4. 配色设计 (适配深蓝 PPT 的高亮色)
# ==========================================
palette = [
    '#7FDBFF',  # Minimal
    '#39CCCC',  # Slight
    '#2ECC40',  # Steady
    '#01FF70',  # Significant
    '#FFDC00',  # High
    '#FF851B',  # Rapid
    '#FF4136',  # Breakthrough
    '#F012BE'  # Extreme
]

# 绘制
bars = sns.barplot(x=counts.index, y=counts.values, palette=palette, edgecolor='white', linewidth=0.8)

# ==========================================
# 5. 细节修饰
# ==========================================
for p in bars.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color='white',
                    xytext=(0, 5), textcoords='offset points')

ax.set_title("Optimization Speedup Distribution(310/467)", fontsize=18, fontweight='bold', pad=20, color='white')
ax.set_ylabel("Number of Test Cases", fontsize=14, color='white')
ax.set_xlabel("Speedup Ratio Range", fontsize=14, color='white', labelpad=10)

ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')
sns.despine(top=True, right=True, left=False, bottom=False)

plt.xticks(rotation=0)
plt.tight_layout()

# 保存透明图片
plt.savefig("blue_ppt_chart_fixed.png", transparent=True, dpi=300)
print("图表已生成 (英文标签版): blue_ppt_chart_fixed.png")
plt.show()