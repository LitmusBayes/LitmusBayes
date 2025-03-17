import json
import os

# ================= 配置输入 =================

litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"

# 假设这是你的数据文件路径
data_file_path = stat_log_base + ".cache4_norm.jsonl"
# 假设这是你的重复分组文件路径
groups_file_path = "identical_vector_groups.json"
# 输出路径
output_file_path = stat_log_base + ".cache4_norm_filter_same.jsonl"


# ================= 1. 准备模拟数据 (你可以跳过这一步直接用你的文件) =================
def create_dummy_files():
    # 模拟你的数据文件 (JSONL格式)
    raw_data = [
        {"litmus": "SB+wsi-rfi-addr+wsi-rfi-ctrl", "param": [1, 5], "score": 0.81},
        {"litmus": "S+rfi-data+addr-rfi-data", "param": [0, 0], "score": 0.16},
        {"litmus": "MP+po+poaqp+NEW", "param": [1, 1], "score": 0.12},
        {"litmus": "2+2W+fence.w.w+rfi-data", "param": [0, 1], "score": 0.62},
        {"litmus": "ISA15", "param": [0, 0], "score": 0.0},  # 应该被删除
        {"litmus": "SB+rfi-addrs", "param": [0, 0], "score": 2.72},  # 应该被删除
        {"litmus": "MP+fence.w.w+data-wsi-rfi-addr", "param": [0, 6], "score": 0.03},  # 应该保留（它是组长）
        {"litmus": "SB+pos-addrs", "param": [0, 0], "score": 2.72}  # 应该保留（它是组长）
    ]

    with open(data_file_path, "w") as f:
        for item in raw_data:
            f.write(json.dumps(item) + "\n")

    # 模拟你的分组文件
    groups_data = [
        [
            "SB+pos-addrs",  # 保留
            "SB+rfi-addrs"  # 删除
        ],
        [
            "MP+fence.w.w+data-wsi-rfi-addr",  # 保留
            "ISA15"  # 删除
        ]
    ]

    with open(groups_file_path, "w") as f:
        json.dump(groups_data, f, indent=4)

    print("--- 模拟文件已生成 ---")


# ================= 2. 核心过滤逻辑 =================

def filter_duplicates(data_path, group_path, out_path):
    # 1. 构建黑名单 (Blocklist)
    print(f"Reading groups from {group_path}...")
    with open(group_path, "r") as f:
        groups = json.load(f)

    blocklist = set()
    for group in groups:
        # group[1:] 表示从第二个元素开始取，直到最后
        # 这些都是要删除的“副本”
        if len(group) > 1:
            duplicates = group[1:]
            blocklist.update(duplicates)

    print(f"Found {len(blocklist)} duplicate names to remove (e.g., {list(blocklist)[:3]}...)")

    # 2. 过滤数据
    print(f"Filtering data from {data_path}...")
    kept_count = 0
    removed_count = 0

    with open(data_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue

            try:
                obj = json.loads(line)
                name = obj.get("litmus")

                # 检查名字是否在黑名单里
                if name in blocklist:
                    removed_count += 1
                    # print(f"Removing duplicate: {name}") # 调试用
                else:
                    fout.write(line + "\n")
                    kept_count += 1

            except json.JSONDecodeError:
                pass

    print("-" * 40)
    print(f"Done! Result saved to: {out_path}")
    print(f"Total Kept:    {kept_count}")
    print(f"Total Removed: {removed_count}")


# ================= 运行 =================

if __name__ == "__main__":
    # 如果你没有真实文件，这一行会生成模拟文件供测试
    # 如果你有真实文件，请注释掉这一行，并修改顶部的路径变量
    # create_dummy_files()

    filter_duplicates(data_file_path, groups_file_path, output_file_path)

    # 打印一下结果看看
    print("\n--- 过滤后的文件内容 ---")
    with open(output_file_path, "r") as f:
        print(f.read())