import json
from collections import defaultdict

input_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_sort.jsonl"
output_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_sort_all.jsonl"

# litmus -> list of records
by_litmus = defaultdict(list)

# param(tuple) -> set(litmus)
param_to_litmus = defaultdict(set)

# 读取并分组
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        by_litmus[obj["litmus"]].append(obj)

for litmus, records in by_litmus.items():
    # 该 litmus 的最大 score
    max_score = max(r["score"] for r in records)

    # 和你原逻辑一致：跳过全 0 的
    if max_score == 0:
        continue

    # 找到所有最优 param
    for r in records:
        if r["score"] == max_score:
            param_key = tuple(r["param"])   # list -> tuple 才能当 dict key
            param_to_litmus[param_key].add(litmus)

# 整理输出结构
output = []
for param, litmus_set in param_to_litmus.items():
    output.append({
        "param": str(param),
        "litmus_count": len(litmus_set),
        "litmus_tests": sorted(litmus_set)
    })

# 按 litmus 数量降序排序（方便你看“最强 param”）
output.sort(key=lambda x: x["litmus_count"], reverse=True)

# 写文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Done.")
print("输出文件:", output_file)
print("param 总数:", len(output))
