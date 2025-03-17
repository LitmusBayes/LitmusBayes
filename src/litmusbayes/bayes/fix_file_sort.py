import os
import json

# 这是你原本获取文件列表的逻辑
directory = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
suffix = ".litmus"  # 举例
files = [
    os.path.join(directory, f) for f in os.listdir(directory)
    if f.endswith(suffix) and os.path.isfile(os.path.join(directory, f))
]

# --- 关键：把这个顺序存成文件 ---
with open("file_order_config.json", "w") as f:
    json.dump(files, f)

print("顺序已备份！请把 file_order_config.json 复制到新电脑。")