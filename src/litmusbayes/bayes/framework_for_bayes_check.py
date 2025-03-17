import json
import logging
import os
import random
import time
from collections import defaultdict
from xgboost import XGBRegressor
from scipy.stats import norm, spearmanr

# 引入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 类定义 =================

class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    key = self._make_key(obj["litmus"], obj["param"])
                    self.data[key] = obj["score"]
        self.f = open(path, "a")

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        pass


class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=200,
                 litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,  # 利用多核
            # max_features="sqrt",
            min_samples_leaf=3,
            random_state=SEED
        )
        # self.model = XGBRegressor(
        #     n_estimators=n_estimators,
        #     learning_rate=0.05,  # 学习率越低越稳，但需要更多 estimator
        #     max_depth=6,  # 树深
        #     subsample=0.8,  # 样本采样
        #     colsample_bytree=0.8,  # 特征采样
        #     n_jobs=-1,
        #     random_state=SEED
        # )
        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)  # 获取 logger

        # 加载向量
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line: continue
                name, vec_str = line.split(":", 1)
                vec = eval(vec_str)
                litmus_to_vec[name] = list(vec)
        return litmus_to_vec

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting...")
        # ============ 关键修改 ============
        # 使用 log1p (log(x+1)) 防止 x=0 报错，同时压缩数值
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    def predict_one(self, litmus_name, param_vec):
        if litmus_name not in self.litmus_to_vector_dict:
            return None
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        feature = list(param_vec) + list(litmus_vec)
        pred_log = self.model.predict([feature])[0]
        # ============ 关键修改 ============
        # 还原回去：exp(x) - 1
        return np.expm1(pred_log)

    def predict_batch(self, litmus_list, param_list):
        """
        批量预测方法
        :param litmus_list: list of litmus names
        :param param_list: list of param vectors
        :return: predictions array, valid_indices (因为有些litmus可能没有vector)
        """
        X_batch = []
        valid_indices = []

        # 1. 构建特征矩阵
        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]
                X_batch.append(list(param) + list(litmus_vec))
                valid_indices.append(i)

        if not X_batch:
            return [], []

        # 2. 批量预测 (一次性调用，速度极大提升)
        # 注意: np.array(X_batch) 依然有开销，但比循环调用 predict 快得多
        X_batch_np = np.array(X_batch)
        pred_log = self.model.predict(X_batch_np)

        # 3. 还原对数
        preds = np.expm1(pred_log)

        return preds, valid_indices

# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
# cache_file_path = stat_log_base + ".cache4_norm_filter_same.jsonl"
# cache_file_path = stat_log_base + ".cache_sum_70_no_norm_gt_0_for_graph.jsonl"

cache_file_path = "log/cache_0.jsonl"
def analyze_ranking_quality(groups_data, output_dir="./analysis_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    regret_list = []
    rank_of_best_list = []

    # 用于绘图的数据
    plot_data = []

    for litmus, records in groups_data.items():
        if len(records) < 2: continue

        # 1. 找到真实的最优值 (Ground Truth Best)
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        gt_best_val = records_sorted_by_actual[0]['actual']

        if gt_best_val <= 0: continue  # 避免除零

        # 2. 找到模型预测的 Top-1 对应的真实值
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        model_pick_val = records_sorted_by_pred[0]['actual']  # 注意：这里取的是被选中那个参数的【真实分】

        # 3. 计算 Regret (性能回撤)
        # 含义：相比于上帝视角的最佳，模型选出的配置慢了百分之多少？
        # 如果是分数越高越好：(Best - Pick) / Best
        regret = (gt_best_val - model_pick_val) / gt_best_val
        regret_list.append(regret)

        # 4. 计算真实最优解在预测列表里的排名 (Rank of Best)
        # 也就是：模型把真正的第一名排到了第几位？
        # 找到真实最佳那个参数在 records_sorted_by_pred 中的索引
        # (可能有多个并列第一，找到任意一个即可)
        best_params_set = set()
        for r in records:
            if r['actual'] == gt_best_val:
                best_params_set.add(tuple(r['param']))

        rank = -1
        for idx, r in enumerate(records_sorted_by_pred):
            if tuple(r['param']) in best_params_set:
                rank = idx + 1  # 排名从1开始
                break

        if rank != -1:
            # 归一化排名 (因为有的测试用例有100个参数，有的只有10个)
            normalized_rank = rank / len(records)
            rank_of_best_list.append(normalized_rank)

    # === 打印诊断报告 ===
    print("\n" + "=" * 40)
    print("      DEEP DIAGNOSIS REPORT      ")
    print("=" * 40)

    regret_arr = np.array(regret_list)
    print(f"Total Groups Analyzed: {len(regret_list)}")
    print(f"Mean Performance Regret: {np.mean(regret_arr) * 100:.2f}%")
    print(f"Median Performance Regret: {np.median(regret_arr) * 100:.2f}%")
    print(f"90th Percentile Regret: {np.percentile(regret_arr, 90) * 100:.2f}%")
    print("-" * 40)
    print(
        f"Zero Regret (Perfect Match): {np.sum(regret_arr == 0)} / {len(regret_arr)} ({np.mean(regret_arr == 0) * 100:.2f}%)")
    print(
        f"Low Regret (<5% loss):       {np.sum(regret_arr < 0.05)} / {len(regret_arr)} ({np.mean(regret_arr < 0.05) * 100:.2f}%)")

    # === 绘图 1: Regret 分布直方图 ===
    plt.figure(figsize=(10, 6))
    sns.histplot(regret_arr, bins=50, kde=True, color='salmon')
    plt.title("Performance Regret Distribution\n(How much performance do we lose by trusting the model?)")
    plt.xlabel("Regret (0.0 = Optimal, 0.1 = 10% worse than optimal)")
    plt.ylabel("Count of Litmus Tests")
    plt.axvline(0.05, color='green', linestyle='--', label='5% Threshold')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "regret_distribution.png"))
    print(f"Plot saved: {os.path.join(output_dir, 'regret_distribution.png')}")

    # === 绘图 2: 真实最优值的排名位置 ===
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(rank_of_best_list)
    plt.title("Where is the Real Best hiding?\n(CDF of Normalized Rank)")
    plt.xlabel("Normalized Rank (0.0 = Top of list, 1.0 = Bottom of list)")
    plt.ylabel("Proportion of Litmus Tests")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "rank_location_cdf.png"))
    print(f"Plot saved: {os.path.join(output_dir, 'rank_location_cdf.png')}")


def deep_error_analysis(groups_data, output_dir="./analysis_plots"):
    logger.info("Starting Deep Error Analysis...")

    # 准备数据容器
    # 每个点代表一个 Litmus Test Group
    group_stats = []

    for litmus, records in groups_data.items():
        if len(records) < 5: continue  # 样本太少不分析

        y_true = np.array([r['actual'] for r in records])
        y_pred = np.array([r['pred'] for r in records])

        # 1. 组内真实分数的变异系数 (CV = Std / Mean)
        # 用来衡量：这组数据本身是否具有区分度？
        # 如果 CV 很小（比如 0.01），说明所有参数跑分都差不多，预测不准很正常。
        cv = np.std(y_true) / (np.mean(y_true) + 1e-6)

        # 2. 组内预测分数的极差 (Range)
        # 用来衡量：模型认为参数对结果的影响有多大？
        pred_range_rel = (np.max(y_pred) - np.min(y_pred)) / (np.mean(y_pred) + 1e-6)

        # 3. Top-1 是否命中
        best_actual = np.max(y_true)
        best_pred_idx = np.argmax(y_pred)
        is_hit = (records[best_pred_idx]['actual'] >= best_actual)

        # 4. 真实最优值的预测排名 (Normalized Rank)
        # 越小越好 (0.0 = 第一名, 1.0 = 最后一名)
        # 找到真实 best 对应的 预测 rank
        sorted_indices = np.argsort(y_pred)[::-1]  # 预测值从大到小
        best_record_indices = [i for i, r in enumerate(records) if r['actual'] == best_actual]

        # 在预测排名里找真实最优
        ranks = []
        for target_idx in best_record_indices:
            # np.where 返回的是 tuple
            r = np.where(sorted_indices == target_idx)[0][0]
            ranks.append(r)

        best_rank_norm = min(ranks) / len(records)

        group_stats.append({
            "litmus": litmus,
            "cv": cv,
            "pred_range_rel": pred_range_rel,
            "is_hit": is_hit,
            "best_rank_norm": best_rank_norm,
            "mean_score": np.mean(y_true)
        })

    df = pd.DataFrame(group_stats)

    # === 分析 1: 数据的区分度 vs 预测准确率 ===
    # 理论上：区分度(CV)越大，模型越容易学到 Top-1
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='cv', y='best_rank_norm', hue='is_hit', alpha=0.6)
    plt.title("Difficulty vs Performance\n(X: Data Variance, Y: Rank of Best (Lower is better))")
    plt.xlabel("Coefficient of Variation (How different are the params?)")
    plt.ylabel("Normalized Rank of True Best (0.0=Top1, 1.0=Last)")
    plt.axvline(x=0.05, color='r', linestyle='--', label='Low Variance Zone')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "diagnosis_variance_vs_rank.png"))

    # === 分析 2: 模型是否躺平 (Model Collapse) ===
    # 检查模型预测的差异范围 (Pred Range) 是否过小
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pred_range_rel', y='best_rank_norm', hue='is_hit', alpha=0.6)
    plt.title("Model Confidence vs Performance\n(X: Predicted Relative Range, Y: Rank Error)")
    plt.xlabel("Predicted Range (Max_Pred - Min_Pred) / Mean")
    plt.ylabel("Normalized Rank of True Best")
    plt.savefig(os.path.join(output_dir, "diagnosis_collapse_check.png"))

    # === 统计输出 ===
    low_var_groups = df[df['cv'] < 0.02]  # 变异系数小于 2% 的组
    high_var_groups = df[df['cv'] >= 0.05]  # 变异系数大于 5% 的组

    print("\n" + "=" * 40)
    print("      ROOT CAUSE DIAGNOSIS      ")
    print("=" * 40)
    print(f"1. Low Variance Groups (Hard to distinguish, CV < 0.02): {len(low_var_groups)}")
    print(f"   -> Top-1 Accuracy: {low_var_groups['is_hit'].mean() * 100:.2f}%")
    print(f"   -> Avg Rank Error: {low_var_groups['best_rank_norm'].mean():.4f}")

    print(f"2. High Variance Groups (Should be easier, CV > 0.05): {len(high_var_groups)}")
    print(f"   -> Top-1 Accuracy: {high_var_groups['is_hit'].mean() * 100:.2f}%")
    print(f"   -> Avg Rank Error: {high_var_groups['best_rank_norm'].mean():.4f}")

    print("-" * 40)
    print("3. Model Boldness Check (Is model predicting flat lines?)")
    flat_prediction_cnt = len(df[df['pred_range_rel'] < 0.01])
    print(
        f"   -> Groups where model predicts <1% difference between Best & Worst param: {flat_prediction_cnt} / {len(df)}")

    if flat_prediction_cnt > len(df) * 0.5:
        print("   >>> WARNING: MODEL COLLAPSE DETECTED. The model is ignoring parameters!")
    else:
        print("   >>> Model is responsive to parameters.")


def analyze_topk_recall(groups_data, output_dir="./analysis_plots"):
    logger.info("Calculating Top-K Recall Curve...")

    # K 的取值点
    k_values = [1, 3, 5, 10, 20, 30, 50]
    hits = {k: 0 for k in k_values}
    total_groups = 0

    for litmus, records in groups_data.items():
        if len(records) < 2: continue
        total_groups += 1

        # 1. 找到真实最优的分数
        max_actual = max(r['actual'] for r in records)

        # 2. 按模型预测排序
        # 注意：这里假设 score 越大越好。如果是越小越好，请改为 reverse=False
        sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)

        # 3. 检查 Top-K 是否包含真实最优
        for k in k_values:
            # 取出前 K 个候选
            candidates = sorted_by_pred[:k]
            # 只要有一个候选的实际分数等于真实最优，就算命中
            # (注意：允许并列第一)
            if any(c['actual'] >= max_actual for c in candidates):
                hits[k] += 1

    # === 输出统计 ===
    print("\n" + "=" * 40)
    print("      TOP-K RECALL ANALYSIS      ")
    print("=" * 40)
    print(f"Total Test Groups: {total_groups}")

    recall_rates = []
    for k in k_values:
        rate = hits[k] / total_groups
        recall_rates.append(rate)
        print(f"Top-{k:<2} Recall: {rate * 100:.2f}%  (Hit: {hits[k]}/{total_groups})")

    # === 绘图 ===
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, recall_rates, marker='o', linewidth=2, color='dodgerblue')
    plt.title("Top-K Recall Curve\n(Probability of finding the True Best within Top-K recommendations)")
    plt.xlabel("K (Number of candidates to verify)")
    plt.ylabel("Recall Rate (Probability)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.yticks(np.arange(0, 1.1, 0.1))

    # 标出 Top-10 的点
    if 10 in k_values:
        idx = k_values.index(10)
        val = recall_rates[idx]
        plt.annotate(f'{val * 100:.1f}%', (10, val), textcoords="offset points", xytext=(0, 10), ha='center')

    save_path = os.path.join(output_dir, "topk_recall_curve.png")
    plt.savefig(save_path)
    print(f"Plot saved: {save_path}")
    print("=" * 40)


def analyze_missed_top3_regret(groups_data):
    missed_regrets = []

    for litmus, records in groups_data.items():
        if len(records) < 2: continue

        # 1. 真实最优
        max_actual = max(r['actual'] for r in records)

        # 2. 预测 Top-3
        sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        top3_preds = sorted_by_pred[:3]

        # 3. 检查是否命中
        hit = any(r['actual'] >= max_actual for r in top3_preds)

        if not hit:
            # 如果没命中，计算 Top-3 里能找到的最好成绩
            best_in_top3 = max(r['actual'] for r in top3_preds)
            # 计算回撤：(真实最优 - Top3最优) / 真实最优
            regret = (max_actual - best_in_top3) / max_actual
            missed_regrets.append(regret)

    print("\n" + "=" * 40)
    print("      MISSED CASE ANALYSIS      ")
    print("=" * 40)
    if len(missed_regrets) > 0:
        arr = np.array(missed_regrets)
        print(f"Total Missed Groups: {len(arr)}")
        print(f"Avg Performance Loss:    {np.mean(arr) * 100:.2f}%")
        print(f"Median Performance Loss: {np.median(arr) * 100:.2f}%")
        print(f"Max Performance Loss:    {np.max(arr) * 100:.2f}%")
        print("-" * 40)
        print(f"Loss < 1% (Good Enough): {np.mean(arr < 0.01) * 100:.2f}% of missed cases")
        print(f"Loss < 5% (Acceptable):  {np.mean(arr < 0.05) * 100:.2f}% of missed cases")
    else:
        print("Amazing! Top-3 Recall is 100%!")
    print("=" * 40)

if __name__ == "__main__":
    # 1. Setup Logger
    random.seed(SEED)
    np.random.seed(SEED)

    # 日志文件命名
    log_file_name = f"{stat_log_base}.check.run.log"
    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Evaluation Run | Seed={SEED} ===")

    # 2. 读取 Litmus List
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=200,
        litmus_vec_path=litmus_vec_path
    )

    # 4. 加载 Cache
    logger.info(f"Loading data from {cache_file_path} ...")
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass
    else:
        logger.error("Cache file not found!")
        exit(1)

    total_count = len(all_data)
    logger.info(f"Total records loaded: {total_count}")

    if total_count <= 10000:
        logger.warning("Data size <= 7000, splitting might be invalid based on request.")

    # 5. 切分数据
    random.shuffle(all_data)
    train_size = int(len(all_data)*0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 构建训练集 & 训练
    logger.info("Building training set...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])

    bo.fit()

    logger.info("Evaluating on test set (Batch Mode)...")

    # 准备批量数据
    test_litmus_names = [item["litmus"] for item in test_data]
    test_params = [item["param"] for item in test_data]
    test_scores = [item["score"] for item in test_data]

    # 使用优化后的批量预测 (如果没有修改类，可以在这里手动构建 X)
    # 这里为了不修改你的类定义太多，我们在外部手动构建 Batch

    X_test = []
    indices_keep = []  # 记录哪些数据是有效的（能找到 vector 的）

    # 1. 快速构建特征 (这一步是纯 Python 列表操作，很快)
    for i, (l_name, p_vec) in enumerate(zip(test_litmus_names, test_params)):
        if l_name in bo.litmus_to_vector_dict:
            vec = bo.litmus_to_vector_dict[l_name]
            X_test.append(list(p_vec) + list(vec))
            indices_keep.append(i)



    logger.info(f"Batch prediction prepared. Valid samples: {len(X_test)}/{len(test_data)}")

    # 2. 执行批量预测 (最关键的加速步骤)
    if len(X_test) > 0:
        # 转换为 numpy array
        X_test_np = np.array(X_test)

        t0 = time.time()
        pred_log = bo.model.predict(X_test_np)  # 只有一次 Python-C 交互
        preds = np.expm1(pred_log)
        logger.info(f"Prediction finished in {time.time() - t0:.4f}s")
    else:
        preds = []

    # 3. 将结果重新映射回 Groups 结构
    groups = defaultdict(list)
    y_true_all = []
    y_pred_all = []

    # 使用 indices_keep 将预测结果对应回原始数据
    for ptr, original_idx in enumerate(indices_keep):
        item = test_data[original_idx]
        litmus = item["litmus"]
        param = item["param"]
        actual_score = item["score"]
        pred_score = preds[ptr]

        record = {
            'param': param,
            'actual': actual_score,
            'pred': pred_score
        }
        groups[litmus].append(record)

        y_true_all.append(actual_score)
        y_pred_all.append(pred_score)

    # 2. 开始针对每个 Litmus Test 进行统计
    total_litmus_cnt = 0  # 有效的 Litmus 文件数 (样本数>1)
    top1_match_cnt = 0  # 预测第一名 = 真实第一名 的次数
    top3_match_cnt = 0  # 真实第一名 在 预测前三名里 的次数

    logger.info("=" * 60)
    logger.info(
        f"{'LITMUS NAME':<30} | {'SAMPLES':<5} | {'TOP-1 MATCH?':<12} | {'ACTUAL BEST':<10} | {'MODEL PICK':<10}")
    logger.info("-" * 60)

    for litmus, records in groups.items():
        # 如果测试集里这个文件只有1条数据，那就没法比大小，跳过
        if len(records) < 2:
            continue

        total_litmus_cnt += 1

        # A. 找出【真实】分数最高的记录 (可能有多个并列最高，取第一个即可，或者逻辑上只要分数达到最高就算对)
        # 按 actual 从大到小排序
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        best_actual_record = records_sorted_by_actual[0]
        max_actual_score = best_actual_record['actual']  # 真实能跑到的最高分

        # B. 找出【模型预测】分数最高的记录
        # 按 pred 从大到小排序
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        best_pred_record = records_sorted_by_pred[0]  # 模型推荐去跑这个
        if max_actual_score == 1:
            total_litmus_cnt -= 1
            continue
        # C. 判定 Top-1 是否命中
        # 判定标准：模型推荐的那个参数，它的【真实分数】是否等于【该组数据的最大真实分数】
        # (这样可以兼容有多个参数并列第一的情况)
        is_top1_correct = (best_pred_record['actual'] >= max_actual_score)

        if is_top1_correct:
            top1_match_cnt += 1
            match_str = "YES"
        else:
            match_str = "NO"

        # D. 判定 Top-3 是否命中 (容错)
        # 看看真实最高分的那个参数，是否出现在了模型预测列表的前3名里
        # 注意：这里我们要找“真实最好的那个参数”是否被模型排到了前三
        # 简化逻辑：模型推荐的前三个里，有没有一个能达到真实最高分的？
        top3_preds = records_sorted_by_pred[:3]
        is_top3_correct = any(r['actual'] >= max_actual_score for r in top3_preds)

        if is_top3_correct:
            top3_match_cnt += 1

        # 打印日志 (只打印前10个或者打印错误的，防止刷屏)
        # 这里为了演示，打印所有有效组
        logger.info(
            f"{litmus[:30]:<30} | {len(records):<5} | {match_str:<12} | {max_actual_score:<10.2f} | {best_pred_record['actual']:<10.2f}")

    # 3. 汇总结果
    if total_litmus_cnt > 0:
        top1_acc = top1_match_cnt / total_litmus_cnt
        top3_acc = top3_match_cnt / total_litmus_cnt

        logger.info("=" * 60)
        logger.info("       PER-LITMUS RANKING RESULTS       ")
        logger.info("=" * 60)
        logger.info(f"Total Unique Litmus Tests: {total_litmus_cnt}")
        logger.info(f"Top-1 Accuracy:          {top1_acc * 100:.2f}% ({top1_match_cnt}/{total_litmus_cnt})")
        logger.info(f"Top-3 Recall:            {top3_acc * 100:.2f}% ({top3_match_cnt}/{total_litmus_cnt})")
        logger.info("-" * 60)

        # 也可以顺便算一下全局的 Rho，作为辅助
        y_true_all = np.array(y_true_all).reshape(-1)
        y_pred_all = np.array(y_pred_all).reshape(-1)
        res = spearmanr(y_true_all, y_pred_all)
        rho = res.statistic if hasattr(res, 'statistic') else res[0]
        logger.info(f"Global Spearman Rho:     {rho:.4f}")

        # === 【新增代码】在这里计算并打印 R^2 ===
        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)  # 既然引入了也可以顺便打出来
        logger.info(f"Global R^2 Score:        {r2:.4f}")
        logger.info(f"Global MAE:              {mae:.4f}")
        # ======================================
        logger.info("=" * 60)

    else:
        logger.warning("No litmus test groups with >1 samples found in test set.")
    # ==========================================
    # 计算更真实的指标：平均组内 Rho (Mean Per-Litmus Rho)
    # ==========================================
    per_litmus_rhos = []

    logger.info("-" * 60)
    logger.info("Calculating Per-Litmus Spearman Correlation...")

    for litmus, records in groups.items():
        # 样本太少没法算相关性
        if len(records) < 3:
            continue

        y_true_local = [r['actual'] for r in records]
        y_pred_local = [r['pred'] for r in records]

        # 如果所有真实值都一样（比如都是0），或者预测值都一样，相关性未定义
        if len(set(y_true_local)) <= 1 or len(set(y_pred_local)) <= 1:
            continue

        # 计算单组的 Rho
        rho_local, _ = spearmanr(y_true_local, y_pred_local)

        # 排除 NaN
        if not np.isnan(rho_local):
            per_litmus_rhos.append(rho_local)

    if per_litmus_rhos:
        mean_rho = np.mean(per_litmus_rhos)
        median_rho = np.median(per_litmus_rhos)
        logger.info(f"Mean Per-Litmus Rho:   {mean_rho:.4f}")
        logger.info(f"Median Per-Litmus Rho: {median_rho:.4f}")
    else:
        logger.warning("Not enough data to calculate per-litmus Rho.")

    logger.info("=" * 60)
    analyze_ranking_quality(groups)
    deep_error_analysis(groups, output_dir="./analysis_plots")

    analyze_topk_recall(groups, output_dir="./analysis_plots")
    analyze_missed_top3_regret(groups)