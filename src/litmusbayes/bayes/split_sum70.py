import json

input_file = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70.jsonl"          # 原始文件
output_no_minus1 = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no.jsonl"
output_only_minus1 = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_filter.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_no_minus1, "w", encoding="utf-8") as fout_no, \
     open(output_only_minus1, "w", encoding="utf-8") as fout_only:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        if obj.get("score") == -1:
            fout_only.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            fout_no.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Done!")
print(f"score != -1 -> {output_no_minus1}")
print(f"score == -1 -> {output_only_minus1}")
