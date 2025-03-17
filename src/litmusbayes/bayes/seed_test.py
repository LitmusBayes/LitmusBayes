import random
import numpy as np
import sys

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

print(f"Python 版本: {sys.version.split()[0]}")
print(f"Numpy 版本: {np.__version__}")

# 测试1: 纯 Python 随机数
r1 = [random.random() for _ in range(3)]
print(f"Random test: {r1}")

# 测试2: Numpy 简单随机数
n1 = np.random.rand(3)
print(f"Numpy rand:  {n1}")

# 测试3: Numpy 复杂分布 (容易受版本影响)
n2 = np.random.normal(0, 1, 3)
print(f"Numpy normal:{n2}")