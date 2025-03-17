import json
import logging
import os
import random
import time
from collections import defaultdict

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import joblib

# 引入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 引入解释性工具
try:
    import shap
except ImportError:
    print("Warning: 'shap' library not found. Please run 'pip install shap' for feature attribution.")
    shap = None

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 类定义 =================

class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=200,
                 litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,  # 利用多核
            min_samples_leaf=3,
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)

        # 加载向量
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line: continue
                    name, vec_str = line.split(":", 1)
                    try:
                        vec = eval(vec_str)
                        litmus_to_vec[name] = list(vec)
                    except:
                        pass
        return litmus_to_vec

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        # 特征顺序：Param (11维) + LitmusVec (N维)
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting...")
        # 使用 log1p (log(x+1)) 防止 x=0 报错，同时压缩数值
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    # ================= 模型保存与加载 =================
    def save_model(self, path):
        self.logger.info(f"Saving model to {path}...")
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.logger.info(f"Loading model from {path}...")
        if os.path.exists(path):
            self.model = joblib.load(path)
            return True
        else:
            self.logger.error(f"Model file {path} not found!")
            return False


# ================= 归因分析函数 =================
def explain_prediction(bo_model, litmus_name, param_vec):
    """
    输入一个 Litmus Test 和一组参数，输出：
    1. 预测分数
    2. SHAP值（每个参数对分数的贡献度）
    """
    logger = get_logger(LOG_NAME)

    # 1. 检查数据
    if litmus_name not in bo_model.litmus_to_vector_dict:
        logger.error(f"Litmus vector for {litmus_name} not found!")
        return

    # 2. 构建特征向量 (Param + LitmusVec)
    litmus_vec = bo_model.litmus_to_vector_dict[litmus_name]
    feature_vec = list(param_vec) + list(litmus_vec)
    X_input = np.array([feature_vec])

    # 3. 进行预测 (Log空间)
    pred_log = bo_model.model.predict(X_input)[0]
    pred_score = np.expm1(pred_log)

    print("\n" + "=" * 60)
    print(f"      PREDICTION ANALYSIS: {litmus_name}      ")
    print("=" * 60)
    print(f"Input Params: {param_vec}")
    print(f"Predicted Score: {pred_score:.4f} (Log: {pred_log:.4f})")
    print("-" * 60)

    # 4. SHAP 归因分析
    if shap is None:
        return

    # 初始化 Explainer
    explainer = shap.TreeExplainer(bo_model.model)
    shap_values = explainer.shap_values(X_input)

    # 获取 Base Value 并确保它是标量 float
    base_value = explainer.expected_value
    if isinstance(base_value, list) or (isinstance(base_value, np.ndarray) and base_value.ndim > 0):
        base_value = base_value[0]
    base_value = float(base_value)

    print(f"Base Value (Avg Log Score): {base_value:.4f}")
    print(f"{'PARAMETER':<15} | {'VALUE':<10} | {'CONTRIBUTION (Log)':<20} | {'IMPACT'}")
    print("-" * 60)

    # 假设 Param 是前 N 个特征
    num_params = len(param_vec)

    # 我们只打印 参数部分 的贡献
    for i in range(num_params):
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        contrib = sv[0][i]  # sv[0] 是第一个样本
        val = param_vec[i]

        # 可视化柱状条
        bar_len = int(abs(contrib) * 20)
        bar = "+" * bar_len if contrib > 0 else "-" * bar_len
        impact = "POSITIVE" if contrib > 0 else "NEGATIVE"
        if abs(contrib) < 0.001: impact = "NEUTRAL"

        print(f"Param_{i:<9} | {val:<10} | {contrib:+.4f} {' ':<13} | {bar}")

    print("-" * 60)
    print("Note: Positive contribution means this parameter value INCREASED the score.")
    print("      Negative contribution means it DECREASED the score.")
    print("=" * 60 + "\n")


def analyze_param_trend(bo_model, param_index, param_name="Param"):
    """
    分析某个具体参数在所有训练数据上的全局趋势
    """
    import shap

    # 1. 准备数据：我们需要模型训练时的所有 X 数据
    # bo_model.X 是 list，需要转为 numpy array
    X_train = np.array(bo_model.X)

    # 2. 计算所有数据的 SHAP 值 (这可能需要一点时间，视数据量而定)
    # TreeExplainer 对随机森林非常快
    explainer = shap.TreeExplainer(bo_model.model)
    shap_values = explainer.shap_values(X_train)

    # 3. 处理 shap_values 的维度问题
    # 如果是回归问题，shap_values 通常是 (n_samples, n_features)
    # 有些版本可能是 list，取 [0]
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # 4. 绘制依赖图 (Dependence Plot)
    # x轴: 参数的实际值
    # y轴: 该参数对预测结果的 SHAP 贡献值
    print(f"Generating dependence plot for feature index {param_index}...")

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        ind=param_index,
        shap_values=shap_values,
        features=X_train,
        feature_names=[f"Param_{i}" for i in range(X_train.shape[1])],  # 如果不指定名字
        interaction_index=None,  # 如果设为 "auto"，它会自动寻找和这个参数交互最强的另一个参数并上色
        show=False
    )
    plt.title(f"Impact of {param_name} (Index {param_index}) on Score")
    plt.tight_layout()
    plt.show()  # 或者 plt.savefig(f"trend_{param_index}.png")

# ================= 主程序 =================
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"

# 指定模型保存的路径 (你可以改成固定的名字，方便重复使用)
MODEL_PATH = "rf_model_trained.pkl"

if __name__ == "__main__":
    # 1. 基础设置
    random.seed(SEED)
    np.random.seed(SEED)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{stat_log_base}.{ts}.shap_analysis.log"

    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )

    # 2. 读取 Litmus 文件列表 (初始化 BO 需要用)
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=100,
        litmus_vec_path=litmus_vec_path
    )

    # ================= 核心修改逻辑 =================

    # 尝试加载模型
    if os.path.exists(MODEL_PATH):
        logger.info(f"Found existing model at {MODEL_PATH}, loading...")
        bo.load_model(MODEL_PATH)
        logger.info("Model loaded successfully. Skipping training.")
    else:
        logger.info(f"Model not found at {MODEL_PATH}. Starting training...")

        # 4. 加载训练数据
        logger.info(f"Loading training data from {cache_file_path} ...")
        all_data = []
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            all_data.append(obj)
                        except:
                            pass
        else:
            logger.error("Cache file not found!")
            exit(1)

        # 5. 训练模型
        logger.info("Building dataset and fitting model...")
        for item in all_data:
            bo.add(item["litmus"], item["param"], item["score"])

        t_start = time.time()
        bo.fit()
        logger.info(f"Model training finished in {time.time() - t_start:.2f}s")

        # 保存模型供下次使用
        bo.save_model(MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

    # ================= 步骤 7: 交互式/演示解释 =================

    logger.info("Running Feature Contribution Analysis (SHAP)...")

    # 在这里输入你想分析的 Case
    target_litmus = "2+2W+rfi-addrs"
    target_param = [1, 5, 1, 1, 0, 1, 4, 2, 0, 0, 0]

    explain_prediction(bo, target_litmus, target_param)

    logger.info("Analysis Done.")

