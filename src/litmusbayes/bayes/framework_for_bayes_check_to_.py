import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from collections import Counter

# ================= 配置 =================
# 你的路径
cache_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_norm_for_graph.jsonl"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"


def diagnose():
    print("=== 开始模型诊断 ===")

    # 1. 加载 Litmus Vector
    litmus_to_vec = {}
    with open(litmus_vec_path, "r") as f:
        for line in f:
            if ":" in line:
                n, v = line.strip().split(":", 1)
                litmus_to_vec[n] = eval(v)

    # 2. 加载数据
    data = []
    scores = []
    litmus_names = []

    with open(cache_file_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                l = obj['litmus']
                if l in litmus_to_vec:
                    data.append(obj)
                    scores.append(obj['score'])
                    litmus_names.append(l)
            except:
                pass

    scores = np.array(scores)
    print(f"\n[数据概览]")
    print(f"总样本数: {len(data)}")
    print(f"覆盖文件数: {len(set(litmus_names))}")
    print(f"分数范围: Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}")
    print(f"0分(或极低分<1e-6)样本占比: {np.sum(scores < 1e-6) / len(scores) * 100:.2f}%")

    # 3. 欠拟合测试
    print(f"\n[模型能力测试]")

    # 准备 X, y
    X = []
    y = []
    for obj in data:
        vec = litmus_to_vec[obj['litmus']]
        X.append(list(obj['param']) + list(vec))
        y.append(obj['score'])

    X = np.array(X)
    y = np.array(y)

    # 切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

    # 对比两组参数
    # A组：你现在的参数 (欠拟合组)
    model_bad = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10,  # 你的原参数
        random_state=2025,
        n_jobs=-1
    )

    # B组：修正后的参数 (过拟合/强力组)
    model_good = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=1,  # 允许生长到底
        max_depth=None,
        random_state=2025,
        n_jobs=-1
    )

    print("\n>>> 正在训练 Model A (min_samples_leaf=10)...")
    model_bad.fit(X_train, y_train)
    train_rho_bad = spearmanr(y_train, model_bad.predict(X_train)).statistic
    test_rho_bad = spearmanr(y_test, model_bad.predict(X_test)).statistic

    print("\n>>> 正在训练 Model B (min_samples_leaf=1)...")
    model_good.fit(X_train, y_train)
    train_rho_good = spearmanr(y_train, model_good.predict(X_train)).statistic
    test_rho_good = spearmanr(y_test, model_good.predict(X_test)).statistic

    print("-" * 60)
    print(f"{'Metric':<20} | {'Model A (Current)':<20} | {'Model B (Fixed)':<20}")
    print("-" * 60)
    print(
        f"{'Train Rho (Fitting)':<20} | {train_rho_bad:.4f}{' (Too Low!)' if train_rho_bad < 0.9 else '' :<15} | {train_rho_good:.4f}")
    print(
        f"{'Test Rho (General)':<20} | {test_rho_bad:.4f} {'(Poor)' if test_rho_bad < 0.5 else '' :<15} | {test_rho_good:.4f}")
    print("-" * 60)

    if train_rho_bad < 0.8:
        print("\n!! 诊断结论: 模型 A 在训练集上表现都很差，确认为【欠拟合】。")
        print("   原因: min_samples_leaf=10 太大了，限制了树的学习能力。")

    # 4. 特征重要性
    print(f"\n[特征重要性 Top 10]")
    importances = model_good.feature_importances_
    # 假设前11位是参数，后面是 Litmus Vec
    # 这里我们只简单打印索引
    indices = np.argsort(importances)[::-1]
    for f in range(10):
        idx = indices[f]
        if idx < 11:
            print(f"Top {f + 1}: Param_Index_{idx} (Score: {importances[idx]:.4f})")
        else:
            print(f"Top {f + 1}: Litmus_Vec_{idx - 11} (Score: {importances[idx]:.4f})")

    # 5. 检查 Litmus Vector 是否起作用
    vec_imp_sum = sum(importances[11:])
    param_imp_sum = sum(importances[:11])
    print(f"\n特征贡献比例: Params={param_imp_sum:.2f}, LitmusVec={vec_imp_sum:.2f}")
    if vec_imp_sum < 0.1:
        print("!! 警告: Litmus Vector 几乎没起作用，模型可能把所有文件混在一起无法区分。")


if __name__ == "__main__":
    diagnose()