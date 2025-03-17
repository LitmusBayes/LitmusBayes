import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 引入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

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


class InteractionRandomForestBO:
    """
    带有显式特征交互 (Explicit Interaction) 的随机森林模型。
    用于解决模型只关注 Litmus 类型而忽略 Param 调整的问题。
    """

    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=500,
                 litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"):
        self.ps = param_space

        # 增加树的数量(500)，因为特征维度增加了
        # min_samples_leaf=2 允许捕捉更细微的 Top-1 差异
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            min_samples_leaf=2,
            random_state=SEED,
            # max_features=0.5 # 可选：强制树在分裂时只看一部分特征，增加多样性
        )

        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)

        # 加载向量
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

        # 特征标准化器 (对于交叉特征很有用)
        self.scaler = StandardScaler()

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if not os.path.exists(path):
            return {}
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

    def _compute_features(self, param_vec, litmus_vec):
        """
        核心逻辑：计算基础特征 + 显式交叉特征
        """
        p = np.array(param_vec)
        l = np.array(litmus_vec)

        # 1. 基础特征 (拼接)
        base_features = list(p) + list(l)

        # 2. 交叉特征 (外积展平)
        # 这捕捉了 "参数P 在 环境L 下" 的特定效果
        #
        interaction = np.outer(p, l).flatten().tolist()

        return base_features + interaction

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]

        # 使用新的特征构造逻辑
        features = self._compute_features(param_vec, litmus_vec)

        self.X.append(features)
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting Interaction RF (Features dim: {len(self.X[0]) if self.X else 0})...")

        X_np = np.array(self.X)
        # 使用 log1p 平滑 Target
        y_train_log = np.log1p(np.array(self.y))

        # 标准化特征
        self.X_scaled = self.scaler.fit_transform(X_np)

        self.model.fit(self.X_scaled, y_train_log)
        self.logger.info("Fitting done.")

    def predict_batch(self, litmus_list, param_list):
        """
        批量预测，内部处理特征构造和标准化
        """
        X_batch = []
        valid_indices = []

        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]

                # 构造同样的交叉特征
                features = self._compute_features(param, litmus_vec)

                X_batch.append(features)
                valid_indices.append(i)

        if not X_batch:
            return [], []

        X_batch_np = np.array(X_batch)

        # 使用训练时拟合的 scaler 进行转换
        X_batch_scaled = self.scaler.transform(X_batch_np)

        pred_log = self.model.predict(X_batch_scaled)

        # 还原对数
        preds = np.expm1(pred_log)

        return preds, valid_indices


# ================= 诊断分析函数 =================

def analyze_ranking_quality(groups_data, output_dir="./analysis_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    regret_list = []
    rank_of_best_list = []

    for litmus, records in groups_data.items():
        if len(records) < 2: continue

        # 1. 真实最优
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        gt_best_val = records_sorted_by_actual[0]['actual']
        if gt_best_val <= 0: continue

        # 2. 模型Top-1
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        model_pick_val = records_sorted_by_pred[0]['actual']

        # 3. Regret
        regret = (gt_best_val - model_pick_val) / gt_best_val
        regret_list.append(regret)

        # 4. Rank of Best
        best_params_set = set()
        for r in records:
            if r['actual'] == gt_best_val:
                best_params_set.add(tuple(r['param']))

        rank = -1
        for idx, r in enumerate(records_sorted_by_pred):
            if tuple(r['param']) in best_params_set:
                rank = idx + 1
                break

        if rank != -1:
            normalized_rank = rank / len(records)
            rank_of_best_list.append(normalized_rank)

    print("\n" + "=" * 40)
    print("      DEEP DIAGNOSIS REPORT      ")
    print("=" * 40)
    regret_arr = np.array(regret_list)
    print(f"Total Groups Analyzed: {len(regret_list)}")
    print(f"Mean Performance Regret: {np.mean(regret_arr) * 100:.2f}%")
    print(f"Median Performance Regret: {np.median(regret_arr) * 100:.2f}%")
    print("-" * 40)
    print(f"Zero Regret (Perfect Match): {np.sum(regret_arr == 0)} ({np.mean(regret_arr == 0) * 100:.2f}%)")
    print(f"Low Regret (<5% loss):       {np.sum(regret_arr < 0.05)} ({np.mean(regret_arr < 0.05) * 100:.2f}%)")

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.histplot(regret_arr, bins=50, kde=True, color='salmon')
    plt.title("Performance Regret Distribution")
    plt.xlabel("Regret (0.0 = Optimal)")
    plt.savefig(os.path.join(output_dir, "regret_distribution.png"))
    print(f"Plot saved: {os.path.join(output_dir, 'regret_distribution.png')}")

    plt.figure(figsize=(10, 6))
    sns.ecdfplot(rank_of_best_list)
    plt.title("CDF of Normalized Rank of Best")
    plt.xlabel("Normalized Rank")
    plt.savefig(os.path.join(output_dir, "rank_location_cdf.png"))
    print(f"Plot saved: {os.path.join(output_dir, 'rank_location_cdf.png')}")


def deep_error_analysis(groups_data, output_dir="./analysis_plots"):
    logger = get_logger(LOG_NAME)
    logger.info("Starting Deep Error Analysis...")
    group_stats = []

    for litmus, records in groups_data.items():
        if len(records) < 5: continue
        y_true = np.array([r['actual'] for r in records])
        y_pred = np.array([r['pred'] for r in records])

        cv = np.std(y_true) / (np.mean(y_true) + 1e-6)
        pred_range_rel = (np.max(y_pred) - np.min(y_pred)) / (np.mean(y_pred) + 1e-6)

        best_actual = np.max(y_true)
        best_pred_idx = np.argmax(y_pred)
        is_hit = (records[best_pred_idx]['actual'] >= best_actual)

        sorted_indices = np.argsort(y_pred)[::-1]
        best_record_indices = [i for i, r in enumerate(records) if r['actual'] == best_actual]
        ranks = [np.where(sorted_indices == target_idx)[0][0] for target_idx in best_record_indices]
        best_rank_norm = min(ranks) / len(records)

        group_stats.append({
            "litmus": litmus, "cv": cv, "pred_range_rel": pred_range_rel,
            "is_hit": is_hit, "best_rank_norm": best_rank_norm
        })

    df = pd.DataFrame(group_stats)

    print("\n" + "=" * 40)
    print("      ROOT CAUSE DIAGNOSIS      ")
    print("=" * 40)
    high_var_groups = df[df['cv'] >= 0.05]
    print(f"High Variance Groups (CV > 0.05): {len(high_var_groups)}")
    if len(high_var_groups) > 0:
        print(f"   -> Top-1 Accuracy: {high_var_groups['is_hit'].mean() * 100:.2f}%")
        print(f"   -> Avg Rank Error: {high_var_groups['best_rank_norm'].mean():.4f}")

    flat_cnt = len(df[df['pred_range_rel'] < 0.01])
    print(f"Flat Prediction Groups: {flat_cnt} / {len(df)}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='cv', y='best_rank_norm', hue='is_hit', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "diagnosis_variance_vs_rank.png"))


def analyze_topk_recall(groups_data, output_dir="./analysis_plots"):
    k_values = [1, 3, 5, 10]
    hits = {k: 0 for k in k_values}
    total_groups = 0

    for litmus, records in groups_data.items():
        if len(records) < 2: continue
        total_groups += 1
        max_actual = max(r['actual'] for r in records)
        sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)

        for k in k_values:
            candidates = sorted_by_pred[:k]
            if any(c['actual'] >= max_actual for c in candidates):
                hits[k] += 1

    print("\n" + "=" * 40)
    print("      TOP-K RECALL ANALYSIS      ")
    print("=" * 40)
    for k in k_values:
        print(f"Top-{k:<2} Recall: {hits[k] / total_groups * 100:.2f}%")


def analyze_missed_top3_regret(groups_data):
    missed_regrets = []
    for litmus, records in groups_data.items():
        if len(records) < 2: continue
        max_actual = max(r['actual'] for r in records)
        sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        top3_preds = sorted_by_pred[:3]

        if not any(r['actual'] >= max_actual for r in top3_preds):
            best_in_top3 = max(r['actual'] for r in top3_preds)
            regret = (max_actual - best_in_top3) / max_actual
            missed_regrets.append(regret)

    print("\n" + "=" * 40)
    print("      MISSED CASE ANALYSIS (Top-3)      ")
    print("=" * 40)
    if missed_regrets:
        arr = np.array(missed_regrets)
        print(f"Missed Count: {len(arr)}")
        print(f"Median Perf Loss: {np.median(arr) * 100:.2f}%")
        print(f"Loss < 5% Cases:  {np.mean(arr < 0.05) * 100:.2f}%")
    else:
        print("No missed cases!")
    print("=" * 40)


# ================= 主程序 =================
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"


if __name__ == "__main__":
    # 1. Setup
    random.seed(SEED)
    np.random.seed(SEED)
    log_file_name = f"{stat_log_base}.interaction_rf.run.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Evaluation Run (Interaction RF) | Seed={SEED} ===")

    # 2. Litmus List
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 InteractionRandomForestBO
    param_space = LitmusParamSpace()
    bo = InteractionRandomForestBO(
        param_space,
        litmus_names,
        n_estimators=500,  # 增加树数量
        litmus_vec_path=litmus_vec_path
    )

    # 4. 加载数据
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

    logger.info(f"Total records loaded: {len(all_data)}")

    # 5. 切分数据
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    # 6. 训练
    logger.info("Building training set...")
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])

    bo.fit()

    # 7. 预测
    logger.info("Evaluating on test set (Batch Mode)...")
    test_litmus_names = [item["litmus"] for item in test_data]
    test_params = [item["param"] for item in test_data]

    # 关键修改：直接调用 bo.predict_batch，不在外部手动构建特征
    t0 = time.time()
    preds, indices_keep = bo.predict_batch(test_litmus_names, test_params)
    logger.info(f"Prediction finished in {time.time() - t0:.4f}s. Valid samples: {len(preds)}")

    # 8. 映射结果
    groups = defaultdict(list)
    y_true_all = []
    y_pred_all = []

    for ptr, original_idx in enumerate(indices_keep):
        item = test_data[original_idx]
        actual_score = item["score"]
        pred_score = preds[ptr]

        record = {
            'param': item["param"],
            'actual': actual_score,
            'pred': pred_score
        }
        groups[item["litmus"]].append(record)

        y_true_all.append(actual_score)
        y_pred_all.append(pred_score)

    # 9. 基础统计
    total_litmus_cnt = 0
    top1_match_cnt = 0
    top3_match_cnt = 0

    logger.info("=" * 60)
    logger.info(
        f"{'LITMUS NAME':<30} | {'SAMPLES':<5} | {'TOP-1 MATCH?':<12} | {'ACTUAL BEST':<10} | {'MODEL PICK':<10}")
    logger.info("-" * 60)

    for litmus, records in groups.items():
        if len(records) < 2: continue
        total_litmus_cnt += 1

        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        max_actual_score = records_sorted_by_actual[0]['actual']

        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        best_pred_record = records_sorted_by_pred[0]

        if max_actual_score == 1:
            total_litmus_cnt -= 1
            continue

        if best_pred_record['actual'] >= max_actual_score:
            top1_match_cnt += 1
            match_str = "YES"
        else:
            match_str = "NO"

        if any(r['actual'] >= max_actual_score for r in records_sorted_by_pred[:3]):
            top3_match_cnt += 1

        logger.info(
            f"{litmus[:30]:<30} | {len(records):<5} | {match_str:<12} | {max_actual_score:<10.2f} | {best_pred_record['actual']:<10.2f}")

    if total_litmus_cnt > 0:
        logger.info("=" * 60)
        logger.info("       PER-LITMUS RANKING RESULTS       ")
        logger.info("=" * 60)
        logger.info(f"Top-1 Accuracy: {top1_match_cnt / total_litmus_cnt * 100:.2f}%")
        logger.info(f"Top-3 Recall:   {top3_match_cnt / total_litmus_cnt * 100:.2f}%")

        r2 = r2_score(y_true_all, y_pred_all)
        res = spearmanr(y_true_all, y_pred_all)
        rho = res.statistic if hasattr(res, 'statistic') else res[0]
        logger.info(f"Global R^2: {r2:.4f}, Global Rho: {rho:.4f}")

    # 10. 深度诊断
    analyze_ranking_quality(groups)
    deep_error_analysis(groups)
    analyze_topk_recall(groups)
    analyze_missed_top3_regret(groups)