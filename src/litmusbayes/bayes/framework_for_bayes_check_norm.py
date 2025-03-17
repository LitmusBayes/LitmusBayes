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


class RankNormalizedRandomForestBO:
    def __init__(self, param_space, litmus_list, n_estimators=200, litmus_vec_path=None):
        self.ps = param_space
        # 回归 Random Forest，抗噪能力更强
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=5,  # 稍微增加叶子节点最小样本数，防止过拟合噪声
            n_jobs=-1,
            random_state=2025,
            # max_features="sqrt"     # 可选：如果特征很多，开启这个可以增加随机性
        )

        self.X = []
        self.y = []
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

        # === 特征工程：显式特征交叉 ===
        # 将 Param 和 Vector 进行外积 (Outer Product) 展平
        # 假设 Param 维度 M，Vector 维度 N，这会增加 M*N 个特征
        # 这能直接告诉模型：“当 Vector是V1 且 Param是P1 时，效果如何”
        p_vec = np.array(param_vec)
        l_vec = np.array(litmus_vec)

        # 原始特征
        features = list(p_vec) + list(l_vec)

        # 交叉特征 (Interaction)
        # 注意：如果特征维度特别大（比如 Vector > 64），这一步可能太慢，可以注释掉
        # 鉴于你的 log 显示 Vector 只有 16 维，Param 11 维，11*16 = 176，这完全没问题！
        interaction = np.outer(p_vec, l_vec).flatten().tolist()
        features += interaction

        self.X.append(features)
        self.y.append(score)
        self.qids.append(litmus_name)

    def fit(self):
        self.logger.info(f"Start fitting Rank-Normalized Random Forest with Interactions...")

        X_np = np.array(self.X)
        y_np = np.array(self.y)
        qids_np = np.array(self.qids)

        # === Group Rank Normalization (依然保持) ===
        group_indices = defaultdict(list)
        for idx, q in enumerate(qids_np):
            group_indices[q].append(idx)

        y_ranked = np.zeros_like(y_np, dtype=float)

        for q, idxs in group_indices.items():
            idxs = np.array(idxs)
            group_scores = y_np[idxs]
            ranks = np.argsort(np.argsort(group_scores))
            if len(idxs) > 1:
                y_ranked[idxs] = ranks / (len(idxs) - 1)
            else:
                y_ranked[idxs] = 0.5

                # 训练 RF 预测 Ranking Score
        self.model.fit(X_np, y_ranked)
        self.logger.info("Fitting done.")

    def predict_batch(self, litmus_list, param_list):
        X_batch = []
        valid_indices = []

        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]

                # 同样的特征构造
                p_vec = np.array(param)
                l_vec = np.array(litmus_vec)
                features = list(p_vec) + list(l_vec)
                interaction = np.outer(p_vec, l_vec).flatten().tolist()
                features += interaction

                X_batch.append(features)
                valid_indices.append(i)

        if not X_batch:
            return [], []

        X_batch_np = np.array(X_batch)
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

    # 日志文件命名 (标记为 RF + Rank Norm + Interaction)
    log_file_name = f"{stat_log_base}.rf_rank_norm.run.log"
    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start Evaluation Run (RF + Rank Norm + Interaction) | Seed={SEED} ===")

    # 2. 读取 Litmus List
    logger.info("Reading litmus file list...")
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO (使用 RankNormalizedRandomForestBO)
    param_space = LitmusParamSpace()
    bo = RankNormalizedRandomForestBO(
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

    # 5. 切分数据
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 构建训练集 & 训练
    logger.info("Building training set...")
    for item in train_data:
        # 注意：这里假设 score 越大越好。如果是耗时，请传入 -item['score']
        bo.add(item["litmus"], item["param"], item["score"])

    # 训练 (内部会自动进行 特征交叉 和 Rank Normalization)
    bo.fit()

    logger.info("Evaluating on test set (Batch Mode)...")

    # 准备批量数据
    test_litmus_names = [item["litmus"] for item in test_data]
    test_params = [item["param"] for item in test_data]

    # === 关键修改 ===
    # 不要在 main 函数里手动构建 X_test 了，因为特征交叉逻辑很复杂。
    # 直接交给 bo.predict_batch 处理
    t0 = time.time()
    preds, indices_keep = bo.predict_batch(test_litmus_names, test_params)
    logger.info(f"Prediction finished in {time.time() - t0:.4f}s. Valid samples: {len(preds)}")

    # 3. 将结果重新映射回 Groups 结构
    groups = defaultdict(list)
    y_true_all = []  # 真实物理值
    y_pred_rank_all = []  # 预测排名分 (0~1)

    # 使用 indices_keep 将预测结果对应回原始数据
    for ptr, original_idx in enumerate(indices_keep):
        item = test_data[original_idx]
        litmus = item["litmus"]
        param = item["param"]
        actual_score = item["score"]
        pred_rank_score = preds[ptr]  # 这是一个 0~1 的分数

        record = {
            'param': param,
            'actual': actual_score,
            'pred': pred_rank_score
        }
        groups[litmus].append(record)

        y_true_all.append(actual_score)
        y_pred_rank_all.append(pred_rank_score)

    # 2. 开始统计 Ranking 指标
    total_litmus_cnt = 0
    top1_match_cnt = 0
    top3_match_cnt = 0

    logger.info("=" * 60)
    logger.info(
        f"{'LITMUS NAME':<30} | {'SAMPLES':<5} | {'TOP-1 MATCH?':<12} | {'ACTUAL BEST':<10} | {'PRED RANK':<10}")
    logger.info("-" * 60)

    for litmus, records in groups.items():
        if len(records) < 2: continue

        total_litmus_cnt += 1

        # A. 找出【真实】分数最高的
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        max_actual_score = records_sorted_by_actual[0]['actual']

        # B. 找出【模型预测】分数最高的
        # 注意：这里按 pred (Rank Score) 排序
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        best_pred_record = records_sorted_by_pred[0]

        # 过滤无效
        if max_actual_score == 1:
            total_litmus_cnt -= 1
            continue

        # C. 判定 Top-1
        # 模型推荐的那个配置，它的真实分数是否等于该组最大真实分数
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

        logger.info(
            f"{litmus[:30]:<30} | {len(records):<5} | {match_str:<12} | {max_actual_score:<10.2f} | {best_pred_record['pred']:<10.2f}")

    # 3. 汇总结果
    if total_litmus_cnt > 0:
        top1_acc = top1_match_cnt / total_litmus_cnt
        top3_acc = top3_match_cnt / total_litmus_cnt

        logger.info("=" * 60)
        logger.info("       PER-LITMUS RANKING RESULTS (RF + Norm)       ")
        logger.info("=" * 60)
        logger.info(f"Total Unique Litmus Tests: {total_litmus_cnt}")
        logger.info(f"Top-1 Accuracy:          {top1_acc * 100:.2f}% ({top1_match_cnt}/{total_litmus_cnt})")
        logger.info(f"Top-3 Recall:            {top3_acc * 100:.2f}% ({top3_match_cnt}/{total_litmus_cnt})")
        logger.info("-" * 60)

        # Global Rho (Valid)
        res = spearmanr(y_true_all, y_pred_rank_all)
        rho = res.statistic if hasattr(res, 'statistic') else res[0]
        logger.info(f"Global Spearman Rho:     {rho:.4f}")

        # === 移除 R2 和 MAE ===
        # 因为我们预测的是 Rank Score (无量纲)，真实值是物理量，算 R2 没意义
        logger.info("(Skipping R2/MAE as model outputs rank scores)")
        logger.info("=" * 60)

    else:
        logger.warning("No litmus test groups with >1 samples found in test set.")

    # ==========================================
    # 计算更真实的指标：平均组内 Rho
    # ==========================================
    per_litmus_rhos = []
    logger.info("-" * 60)
    logger.info("Calculating Per-Litmus Spearman Correlation...")

    for litmus, records in groups.items():
        if len(records) < 3: continue

        y_true_local = [r['actual'] for r in records]
        y_pred_local = [r['pred'] for r in records]

        if len(set(y_true_local)) <= 1 or len(set(y_pred_local)) <= 1:
            continue

        rho_local, _ = spearmanr(y_true_local, y_pred_local)
        if not np.isnan(rho_local):
            per_litmus_rhos.append(rho_local)

    if per_litmus_rhos:
        logger.info(f"Mean Per-Litmus Rho:   {np.mean(per_litmus_rhos):.4f}")
        logger.info(f"Median Per-Litmus Rho: {np.median(per_litmus_rhos):.4f}")

    logger.info("=" * 60)

    # 诊断分析
    analyze_ranking_quality(groups, output_dir="./analysis_plots_rf_norm")

