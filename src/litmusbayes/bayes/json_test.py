import json

path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector2.jsonl"

with open(path) as f:
    for lineno, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print("❌ Bad line:", lineno)
            print("repr:", repr(line))
            print("error:", e)
            break
