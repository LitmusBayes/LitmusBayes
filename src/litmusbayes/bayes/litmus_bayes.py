from ax import Client, RangeParameterConfig, ChoiceParameterConfig
from ax.service.ax_client import AxClient
experiment_iterations = 1 # Total number of tuning iterations for the experiment





import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# ===========================
# 1. 模拟数据准备
# ===========================

# 假设：
# - 10个可调参数 (Parameters)
# - 5个Litmus Test特征 (Context features)
DIM_PARAMS = 10
DIM_CONTEXT = 5
TOTAL_DIM = DIM_PARAMS + DIM_CONTEXT

# 模拟已有的观测数据 (比如过去的100次运行结果)
# X的结构: [Param_0, ..., Param_9, Context_0, ..., Context_4]
train_X = torch.rand(100, TOTAL_DIM)

# 模拟结果 Y (触发乱序的次数，标准化处理)
# 实际中这是你从仿真器读回来的数值
train_Y = torch.sin(train_X[:, 0] * 10) + train_X[:, -1] # 假函数
train_Y = standardize(train_Y).unsqueeze(-1)

# ===========================
# 2. 构建高斯过程模型 (GP)
# ===========================

# 这里直接使用 SingleTaskGP。
# 虽然我们有多个 Test，但我们把 Test 特征作为输入处理，
# 所以从模型角度看，这是一个定义在高维空间上的单一函数。
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# 拟合模型 (训练超参数)
print("正在拟合 GP 模型...")
fit_gpytorch_mll(mll)
print("模型拟合完成。")

# ===========================
# 3. 为指定的 Litmus Test 推荐参数
# ===========================

# 假设我们要优化的目标是第 500 号 Litmus Test
# 我们先拿到它的特征向量 (这就相当于 Context)
target_litmus_features = torch.rand(DIM_CONTEXT) # 实际中从你的特征库里取

# 定义 Acquisition Function (UCB)
UCB = UpperConfidenceBound(gp, beta=0.1)

# 核心难点：我们需要优化前10个参数，但固定后5个参数(Context)
# BoTorch 的 optimize_acqf 支持 fixed_features
# 格式: {维度索引: 固定值}
fixed_features_dict = {}
for i in range(DIM_CONTEXT):
    # 注意：特征在 X 的末尾，索引是 DIM_PARAMS + i
    fixed_features_dict[DIM_PARAMS + i] = target_litmus_features[i].item()

# 定义参数的搜索边界 (10个参数，范围0-1)
bounds = torch.stack([torch.zeros(DIM_PARAMS), torch.ones(DIM_PARAMS)])

# 开始搜索最优参数
candidate, acq_value = optimize_acqf(
    acq_function=UCB,
    bounds=bounds,
    q=1, # 每次推荐 1 组参数
    num_restarts=5,
    raw_samples=20,
    fixed_features=fixed_features_dict, # <--- 关键：锁定 Litmus 特征
)

# ===========================
# 4. 结果处理
# ===========================

recommended_params = candidate[0, :DIM_PARAMS] # 取出前10位参数
print("\n针对该 Litmus Test 的推荐参数 (Continuous):")
print(recommended_params)

# 如果你的参数必须是 Bool 或 离散值：
# 简单做法：四舍五入
discrete_params = torch.round(recommended_params).int()
print("\n转换为离散/Bool 参数:")
print(discrete_params)

# 下一步：
# 1. 将 discrete_params 和 target_litmus_features 拼起来，作为一次新的 X
# 2. 扔进硅前仿真器运行
# 3. 拿到新的 Y
# 4. train_X.append(new_X), train_Y.append(new_Y) -> 重新训练 Loop