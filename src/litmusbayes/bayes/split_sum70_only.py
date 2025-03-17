import json
import math
from collections import defaultdict

input_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_sort.jsonl"


TARGET_PARAM = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]

# litmus -> list of records
by_litmus = defaultdict(list)
add_list = []
zero = 0
time1 = 0
time2 = 0
# 读取并分组
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        by_litmus[obj["litmus"]].append(obj)

qualified_litmus = []

for litmus, records in by_litmus.items():
    # 该 litmus 的最大 score
    max_score = max(r["score"] for r in records)

    # 该 litmus 在 TARGET_PARAM 下的 score（可能多个，取最大）
    target_scores = [
        r["score"] for r in records
        if r.get("param") == TARGET_PARAM
    ]

    if not target_scores:
        continue

    target_score = max(target_scores)

    if max_score == 0:
        continue
    # 判断是否等于该 litmus 的最大值
    if target_score == max_score:
        qualified_litmus.append(litmus)
    else:
        if target_score == 0:
            zero += 1
            continue
        add_list.append(max_score / target_score)
        item_time1 = 3/target_score
        item_time2 = 3/max_score
        if item_time2 * 3 < item_time1:
            print(f"litmus test : {litmus}, {item_time1}, {item_time2}, speed up: {item_time1/item_time2}")
        time1 += 3/target_score
        time2 += 3/max_score

print("目标 param:", TARGET_PARAM)
print("满足条件的 litmus test 数量:", len(qualified_litmus))
print("litmus test 列表:")
print("len", len(add_list))
print("mean:" ,sum(add_list)/len(add_list))
print("median:" ,add_list[int(len(add_list)/2)])
print("time1:", time1)
print("time2:", time2)
print("zero:" ,zero)
for l in qualified_litmus:
    print(" ", l)
