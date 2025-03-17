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
import xgboost as xgb

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


from sklearn.preprocessing import MinMaxScaler


class RankNormalizedXGBRegressorBO:
    def __init__(self, param_space, litmus_list, n_estimators=500, litmus_vec_path=None):
        self.ps = param_space

        # 使用较深的树来拟合参数交互
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=2025,
            objective='reg:squarederror'
        )

        self.X = []
        self.y = []
        # === 修复点 1: 初始化 qids 列表 ===
        self.qids = []

        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path) if litmus_vec_path else {}

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if not path: return {}
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

        # === 修复点 2: 在添加数据时记录 qid (litmus_name) ===
        self.qids.append(litmus_name)

    def fit(self):
        self.logger.info(f"Start fitting Rank-Normalized XGBRegressor...")

        X_np = np.array(self.X)
        y_np = np.array(self.y)
        qids_np = np.array(self.qids)

        # === 终极魔法：Group Rank Normalization ===
        # 将每个组内的分数转换为 0.0 ~ 1.0 的排名分

        # 1. 按组整理数据索引
        group_indices = defaultdict(list)
        for idx, q in enumerate(qids_np):
            group_indices[q].append(idx)

        # 2. 创建一个新的 y 数组
        y_ranked = np.zeros_like(y_np, dtype=float)

        for q, idxs in group_indices.items():
            idxs = np.array(idxs)
            group_scores = y_np[idxs]

            # 计算排名 (0, 1, 2...)
            # argsort两次可以得到每个元素的排名
            ranks = np.argsort(np.argsort(group_scores))

            # 归一化到 0 ~ 1
            if len(idxs) > 1:
                y_ranked[idxs] = ranks / (len(idxs) - 1)
            else:
                y_ranked[idxs] = 0.5  # 只有一个样本就躺平

        # 3. 训练 (Fit 排名后的 0~1 分数)
        # 此时，Litmus Vector 对预测 y 的均值毫无帮助（因为大家都均匀分布在0-1）
        # 模型必须依赖 Param 来区分 0.9 和 0.1
        self.model.fit(X_np, y_ranked, verbose=False)

        self.logger.info("Fitting done (Rank Normalized).")

    def predict_batch(self, litmus_list, param_list):
        X_batch = []
        valid_indices = []

        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]
                X_batch.append(list(param) + list(litmus_vec))
                valid_indices.append(i)

        if not X_batch:
            return [], []

        X_batch_np = np.array(X_batch)
        # 预测出来的是 "预测排名百分比" (0.9 表示预测这是个好参数)
        preds = self.model.predict(X_batch_np)

        return preds, valid_indices
# ================= 主程序 =================

# 配置路径
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"


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


if __name__ == "__main__":
    # 1. Setup Logger
    random.seed(SEED)
    np.random.seed(SEED)

    # 日志文件命名
    log_file_name = f"{stat_log_base}.xgbrank.run.log"
    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Evaluation Run (XGBRanker) | Seed={SEED} ===")

    # 2. 读取 Litmus List
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO (切换为 XGBoostRankerBO)
    param_space = LitmusParamSpace()
    # 注意：这里使用的是 XGBoostRankerBO
    bo = RankNormalizedXGBRegressorBO(
        param_space,
        litmus_names,
        n_estimators=200,  # 可以根据需要调整，Ranker通常不需要太多树就能收敛
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

    # 5. 切分数据
    # 注意：对于 Ranking 任务，如果目标是泛化到【未知测试用例】，应该按 LitmusName 进行 GroupKFold。
    # 如果目标是【已知测试用例的参数调优】，Random Shuffle 是可以的。
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 构建训练集 & 训练
    logger.info("Building training set...")
    for item in train_data:
        # 确保 item['score'] 是【越大越好】。如果是耗时，请传入 -item['score']
        bo.add(item["litmus"], item["param"], item["score"])

    # 训练模型 (内部会自动按 Group 处理)
    bo.fit()

    logger.info("Evaluating on test set (Batch Mode)...")

    # 准备批量数据
    test_litmus_names = [item["litmus"] for item in test_data]
    test_params = [item["param"] for item in test_data]

    # 1. 执行预测
    # 注意：preds 返回的是 Ranking Score，不是真实性能值
    preds, indices_keep = bo.predict_batch(test_litmus_names, test_params)

    logger.info(f"Prediction finished. Valid samples: {len(preds)}")

    # 2. 将结果重新映射回 Groups 结构
    groups = defaultdict(list)
    y_true_all = []  # 真实值 (绝对值)
    y_pred_rank_scores = []  # 预测值 (排序分)

    for ptr, original_idx in enumerate(indices_keep):
        item = test_data[original_idx]
        litmus = item["litmus"]
        param = item["param"]
        actual_score = item["score"]
        pred_rank_score = preds[ptr]  # 这是一个无量纲的分数

        record = {
            'param': param,
            'actual': actual_score,  # 真实性能
            'pred': pred_rank_score  # 模型给出的排序分
        }
        groups[litmus].append(record)

        y_true_all.append(actual_score)
        y_pred_rank_scores.append(pred_rank_score)

    # 3. 开始统计 Ranking 指标
    total_litmus_cnt = 0
    top1_match_cnt = 0
    top3_match_cnt = 0

    logger.info("=" * 60)
    logger.info(
        f"{'LITMUS NAME':<30} | {'SAMPLES':<5} | {'TOP-1 MATCH?':<12} | {'ACTUAL BEST':<10} | {'RANK SCORE':<10}")
    logger.info("-" * 60)

    for litmus, records in groups.items():
        if len(records) < 2: continue

        total_litmus_cnt += 1

        # A. 找出【真实】分数最高的
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        max_actual_score = records_sorted_by_actual[0]['actual']

        # B. 找出【模型预测】分数最高的 (注意：这里按 pred 排序，pred 是 ranking score)
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        best_pred_record = records_sorted_by_pred[0]

        # 过滤掉全 1 的无效测试用例 (根据你的逻辑)
        if max_actual_score == 1:
            total_litmus_cnt -= 1
            continue

        # C. 判定 Top-1
        # 模型选出来的那个配置，它的真实分数是否等于该组最大真实分数
        is_top1_correct = (best_pred_record['actual'] >= max_actual_score)
        if is_top1_correct:
            top1_match_cnt += 1
            match_str = "YES"
        else:
            match_str = "NO"

        # D. 判定 Top-3
        top3_preds = records_sorted_by_pred[:3]
        is_top3_correct = any(r['actual'] >= max_actual_score for r in top3_preds)
        if is_top3_correct:
            top3_match_cnt += 1

        # 日志 (打印 Rank Score 而不是预测值，因为 Ranker 不预测绝对值)
        logger.info(
            f"{litmus[:30]:<30} | {len(records):<5} | {match_str:<12} | {max_actual_score:<10.2f} | {best_pred_record['pred']:<10.2f}")

    # 4. 汇总结果
    if total_litmus_cnt > 0:
        top1_acc = top1_match_cnt / total_litmus_cnt
        top3_acc = top3_match_cnt / total_litmus_cnt

        logger.info("=" * 60)
        logger.info("       PER-LITMUS RANKING RESULTS (XGBRanker)       ")
        logger.info("=" * 60)
        logger.info(f"Total Unique Litmus Tests: {total_litmus_cnt}")
        logger.info(f"Top-1 Accuracy:          {top1_acc * 100:.2f}% ({top1_match_cnt}/{total_litmus_cnt})")
        logger.info(f"Top-3 Recall:            {top3_acc * 100:.2f}% ({top3_match_cnt}/{total_litmus_cnt})")
        logger.info("-" * 60)

        # Global Rho (Rank Correlation)
        # 注意：这里计算的是 真实值 vs 排序分 的相关性，依然有效
        res = spearmanr(y_true_all, y_pred_rank_scores)
        rho = res.statistic if hasattr(res, 'statistic') else res[0]
        logger.info(f"Global Spearman Rho:     {rho:.4f}")

        # === 重要：移除 R2 和 MAE ===
        # Ranker 输出的是无量纲分数，不能和真实 Label 计算误差
        logger.info("(R2 and MAE are skipped because Ranker outputs dimensionless scores, not absolute values)")
        logger.info("=" * 60)

    else:
        logger.warning("No litmus test groups with >1 samples found in test set.")

    # 5. 计算 Per-Litmus Rho
    per_litmus_rhos = []
    logger.info("-" * 60)
    logger.info("Calculating Per-Litmus Spearman Correlation...")

    for litmus, records in groups.items():
        if len(records) < 3: continue
        y_true_local = [r['actual'] for r in records]
        y_pred_local = [r['pred'] for r in records]  # Rank Scores

        if len(set(y_true_local)) <= 1 or len(set(y_pred_local)) <= 1: continue

        rho_local, _ = spearmanr(y_true_local, y_pred_local)
        if not np.isnan(rho_local):
            per_litmus_rhos.append(rho_local)

    if per_litmus_rhos:
        logger.info(f"Mean Per-Litmus Rho:   {np.mean(per_litmus_rhos):.4f}")
        logger.info(f"Median Per-Litmus Rho: {np.median(per_litmus_rhos):.4f}")
    else:
        logger.warning("Not enough data to calculate per-litmus Rho.")

    logger.info("=" * 60)

    # 6. 调用深度诊断 (计算 Regret)
    # 这步最关键，它会告诉你换了模型后，性能回撤是否减少了
    analyze_ranking_quality(groups, output_dir="./analysis_plots_xgbrank")
    # === 新增：特征重要性分析 ===
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")

    # 假设 param 长度是固定的，我们需要知道 param 的维度
    # 假设 param_space 有 dim 属性，或者我们取第一个样本看看
    if len(all_data) > 0:
        sample_param_len = len(all_data[0]['param'])
        # 剩下的就是 litmus vector 的长度
        total_len = bo.model.feature_importances_.shape[0]
        vec_len = total_len - sample_param_len

        importances = bo.model.feature_importances_

        param_imp_sum = np.sum(importances[:sample_param_len])
        vec_imp_sum = np.sum(importances[sample_param_len:])

        logger.info(f"Total Features: {total_len} (Param: {sample_param_len}, Vector: {vec_len})")
        logger.info(f"Sum Importance - Param Configs:  {param_imp_sum:.4f}")
        logger.info(f"Sum Importance - Litmus Vector:  {vec_imp_sum:.4f}")

        if vec_imp_sum < 0.1:
            logger.warning(
                "!!! DANGER !!! The model is ignoring the Litmus Vector! It is learning a 'Global Average' configuration.")

    logger.info("=" * 60)