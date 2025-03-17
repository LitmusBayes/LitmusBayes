import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略一些pandas的警告
warnings.filterwarnings('ignore')


# ==========================================
# 1. 模拟数据生成 (实际使用时请注释掉这一块，直接加载你的文件)
# ==========================================
def create_dummy_files():
    # 模拟你的70组探索日志 (exploration.jsonl)
    data = [
        {"litmus": "SB+pos-po+pos-po-addrs", "param": [0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0], "score": 3828.5},
        # 模拟一些低分数据作为对比
        {"litmus": "SB+pos-po+pos-po-addrs", "param": [1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0], "score": 120.0},
        {"litmus": "SB+pos-po+pos-po-addrs", "param": [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0], "score": 3500.0},

        {"litmus": "SB+po+pos-po", "param": [1, 1, 0, 0, 2, 0, 2, 1, 0, 1, 0], "score": 3381.6},
        {"litmus": "SB+po+pos-po", "param": [1, 1, 0, 0, 2, 0, 2, 1, 1, 1, 0], "score": 40.5},  # 假设改了第8个参数掉速
    ]
    # 扩充数据以便跑随机森林 (真实场景下你有70条，这里简单复制模拟)
    expanded_data = data * 14

    with open('exploration.jsonl', 'w') as f:
        for entry in expanded_data:
            f.write(json.dumps(entry) + '\n')

    # 模拟预训练模型(表现差)的日志 (bo_model.jsonl)
    bo_data = [
        {"litmus": "SB+pos-po+pos-po-addrs", "param": [1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0], "score": 110.0},
        {"litmus": "SB+po+pos-po", "param": [1, 1, 0, 0, 2, 0, 2, 1, 1, 1, 0], "score": 35.0},
    ]
    with open('bo_model.jsonl', 'w') as f:
        for entry in bo_data:
            f.write(json.dumps(entry) + '\n')


create_dummy_files()
print("模拟文件生成完毕，开始分析...\n")


# ==========================================
# 2. 数据加载与预处理函数
# ==========================================

def load_log_to_df(filepath, source_tag):
    """
    读取jsonl日志并转换为DataFrame，展开param列
    """
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                # 扁平化处理：将param数组拆分为 param_0, param_1 ...
                row = item.copy()
                params = row.pop('param')
                row['score'] = float(row['score'])
                row['source'] = source_tag  # 标记来源：exploration, bo, default

                for i, p_val in enumerate(params):
                    row[f'param_{i}'] = p_val

                data.append(row)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"警告: 文件 {filepath} 未找到。")
        return pd.DataFrame()


# ==========================================
# 3. 核心分析逻辑：随机森林归因
# ==========================================

def analyze_performance_leap(explore_file, bo_file, default_file=None):
    # 1. 加载所有数据
    df_explore = load_log_to_df(explore_file, 'exploration')
    df_bo = load_log_to_df(bo_file, 'bo_bad')

    # 合并主要关注的数据 (探索数据 + 预训练坏数据) 用于对比
    # 我们主要想知道：为什么探索里的某些组合比bo_bad好那么多
    df_all = pd.concat([df_explore, df_bo], ignore_index=True)

    if df_all.empty:
        print("没有数据可分析")
        return

    # 获取参数列名
    param_cols = [c for c in df_all.columns if c.startswith('param_')]
    unique_litmus = df_all['litmus'].unique()

    analysis_results = []

    print(f"{'Litmus Test':<35} | {'Best Score':<10} | {'BO Score':<10} | {'Top Critical Param':<20} | {'Reasoning'}")
    print("-" * 120)

    for litmus in unique_litmus:
        # 获取该测试用例的所有数据
        df_sub = df_all[df_all['litmus'] == litmus].copy()

        # 分离出 BO 的数据 (基准) 和 探索出的最佳数据
        bo_row = df_sub[df_sub['source'] == 'bo_bad']
        best_row = df_sub[df_sub['source'] == 'exploration'].sort_values('score', ascending=False).head(1)

        if bo_row.empty or best_row.empty:
            continue

        bo_score = bo_row.iloc[0]['score']
        best_score = best_row.iloc[0]['score']

        # 如果提升不明显，可能不需要分析 (这里设定阈值，例如提升了50%才分析)
        if best_score < bo_score * 1.2:
            continue

        # --- 归因分析核心：使用随机森林判断哪个参数主导了分数的波动 ---
        X = df_sub[param_cols]
        y = df_sub['score']

        # 训练回归模型
        # n_estimators不需要太多，树深度限制一下防止过拟合
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)

        # 获取特征重要性
        importances = rf.feature_importances_
        # 找出最重要的前2个参数索引
        top_indices = np.argsort(importances)[::-1][:2]

        top_param_name = param_cols[top_indices[0]]
        importance_score = importances[top_indices[0]]

        # 对比该参数在 BO 和 Best 中的值
        val_in_bo = bo_row.iloc[0][top_param_name]
        val_in_best = best_row.iloc[0][top_param_name]

        # 生成结论字符串
        if val_in_bo != val_in_best:
            reason = f"{top_param_name}: {int(val_in_bo)} -> {int(val_in_best)} (Imp: {importance_score:.2f})"
        else:
            # 如果最重要的参数值没变，说明是次要参数或交互作用导致的
            second_param = param_cols[top_indices[1]]
            val2_bo = bo_row.iloc[0][second_param]
            val2_best = best_row.iloc[0][second_param]
            reason = f"Mainly {second_param}: {int(val2_bo)} -> {int(val2_best)} (Comb with {top_param_name})"

        print(f"{litmus:<35} | {best_score:<10.1f} | {bo_score:<10.1f} | {top_param_name:<20} | {reason}")

        analysis_results.append({
            'litmus': litmus,
            'critical_param': top_param_name,
            'best_val': val_in_best,
            'bo_val': val_in_bo,
            'data': df_sub
        })

    return analysis_results


# ==========================================
# 4. 可视化函数：画出关键参数对分数的影响
# ==========================================
def plot_critical_params(analysis_results):
    if not analysis_results:
        return

    # 只画前3个最典型的例子
    for res in analysis_results[:3]:
        litmus = res['litmus']
        param = res['critical_param']
        df = res['data']

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=param, y='score', data=df)
        plt.title(f"Impact of {param} on {litmus}")
        plt.xlabel(f"Value of {param}")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()  # 在notebook或支持图形界面的终端会显示
        print(f"已生成 {litmus} 的参数分布图 (X轴为 {param})")


# ==========================================
# 运行脚本
# ==========================================

# 替换为你真实的文件名
EXPLORE_LOG = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes_banana.log.cache.jsonl'  # 你那70组参数的日志
BO_LOG = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_random.log.validation_cache_banana.jsonl'  # 预训练模型跑出的差结果日志
DEFAULT_LOG = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_init.log.cache.jsonl'  # 默认参数日志 (如有)

# log_path1 = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_init.log.cache.jsonl'
# log_path2 = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes_banana.log.cache.jsonl'


results = analyze_performance_leap(EXPLORE_LOG, BO_LOG)
plot_critical_params(results)