import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

# ===========================
# 1. 维度定义
# ===========================
dim_params = 10   # 参数维度 (0-9)
dim_context = 5   # 特征维度 (10-14)
total_dim = dim_params + dim_context

# 模拟数据
train_X = torch.rand(50, total_dim)
train_Y = torch.rand(50, 1)

# ===========================
# 2. 自定义乘积核 (Product Kernel)
# ===========================

# 定义作用于 [参数部分] 的核
# active_dims=list(range(10)) 告诉核函数只看前10列
covar_param = MaternKernel(
    nu=2.5,
    ard_num_dims=dim_params,
    active_dims=list(range(dim_params)),
    lengthscale_prior=GammaPrior(3.0, 6.0) # 可选：加先验防止过拟合
)

# 定义作用于 [Context 特征部分] 的核
# active_dims 告诉核函数只看后5列 (从第10列到第14列)
covar_context = MaternKernel(
    nu=2.5,
    ard_num_dims=dim_context,
    active_dims=list(range(dim_params, total_dim)),
    lengthscale_prior=GammaPrior(3.0, 6.0)
)

# 核心步骤：将两个核相乘，并包在 ScaleKernel 中以自适应输出幅度
# Math: k_combined = theta * (k_param * k_context)
custom_kernel = ScaleKernel(covar_param * covar_context)

# ===========================
# 3. 初始化 GP 模型
# ===========================

# 传入 covar_module 使用我们需要自定义的核
gp = SingleTaskGP(train_X, train_Y, covar_module=custom_kernel)

# ===========================
# 4. 训练与使用
# ===========================
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

print("模型训练完成，使用了自定义乘积核: k(param) * k(context)")

# 打印一下看看结构
# 你会看到类似于 ScaleKernel(MaternKernel(...) * MaternKernel(...)) 的结构
print(gp.covar_module)