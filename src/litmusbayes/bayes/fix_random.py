import os
import sys

# 把你原来的 get_bounds 函数复制过来
def get_bounds_test():
    # 模拟你原来的列表长度，如果原来的 self.mem_list 长度是动态的，
    # 你可能需要在这里填入当时的近似值，或者直接把原程序跑起来打印 keys
    return {
        "mem": (0, 10), "barrier": (0, 10), "alloc": (0, 10),
        "detached": (0, 1), "thread": (0, 10), "launch": (0, 10),
        "affinity": (0, 10), "stride": (0, 10), "contiguous": (0, 1),
        "noalign": (0, 10), "perple": (0, 0)
    }

print(f"Python Version: {sys.version}")

# 1. 侦查字典顺序
bounds = get_bounds_test()
print("\n=== 也就是这个顺序害了你 ===")
print("旧电脑原本的字典顺序 (Keys):")
print(list(bounds.keys()))
# ↑↑↑ 把这个列表复制下来，这就是你要在新电脑上 hardcode 的 ordered_keys！

# 2. 侦查文件顺序 (如果有用到 os.listdir)
# directory = "你的数据目录路径"
# print("\n=== 文件读取顺序 ===")
# print(os.listdir(directory))