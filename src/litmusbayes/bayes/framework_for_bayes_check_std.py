import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np

# 回退到 RandomForest
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

from src.litmusbayes.bayes.litmus_param_space import LitmusParamSpace
from src.litmusbayes.bayes.logger_util import setup_logger, get_logger
from src.litmusbayes.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 类定义 (保持不变) =================

class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=200,
                 litmus_vec_path=""):
        self.ps = param_space
        # 回归 Random Forest
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            min_samples_leaf=2,  # 保持你原始的参数，防止过拟合
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line: continue
                    try:
                        name, vec_str = line.split(":", 1)
                        vec = eval(vec_str)
                        litmus_to_vec[name] = list(vec)
                    except:
                        pass
        return litmus_to_vec

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec) + list(litmus_vec))
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting RandomForest...")
        if not self.X: return
        # 保持 log1p 处理
        y_train_log = np.log1p(np.array(self.y))
        self.model.fit(np.array(self.X), y_train_log)

    def predict_batch_with_uncertainty(self, litmus_list, param_list):
        """
        核心修改：利用森林里的每棵树来计算方差
        """
        X_batch = []
        valid_indices = []

        # 1. 构建特征
        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]
                X_batch.append(list(param) + list(litmus_vec))
                valid_indices.append(i)

        if not X_batch:
            return [], [], []

        X_batch_np = np.array(X_batch)

        # 2. 获取森林中每一棵树的预测结果
        all_tree_preds_log = []

        for tree in self.model.estimators_:
            preds = tree.predict(X_batch_np)
            all_tree_preds_log.append(preds)

        all_tree_preds_log = np.array(all_tree_preds_log)

        # 3. 还原到真实数值域 (Real Scale)
        all_tree_preds_real = np.expm1(all_tree_preds_log)

        # 4. 计算均值和标准差 (Standard Deviation)
        means = np.mean(all_tree_preds_real, axis=0)
        stds = np.std(all_tree_preds_real, axis=0)

        return means, stds, valid_indices


# ================= 主程序 =================

# 配置路径 (保持不变)
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_gt_0_for_graph.jsonl"

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    # 日志
    log_file_name = f"{stat_log_base}.rf_check.log"
    logger = setup_logger(
        log_file=log_file_name,
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"=== Start RF Evaluation | Seed={SEED} ===")

    # 读取 Litmus List
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 初始化 RF BO
    param_space = LitmusParamSpace()
    bo = RandomForestBO(
        param_space,
        litmus_names,
        n_estimators=200,
        litmus_vec_path=litmus_vec_path
    )

    # 加载数据
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass

    # 切分数据
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    logger.info(f"Train size: {len(train_data)} | Test size: {len(test_data)}")

    # 训练
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])
    bo.fit()

    # 预测 (带不确定性)
    logger.info("Evaluating on test set (Batch Mode)...")
    test_litmus = [item["litmus"] for item in test_data]
    test_params = [item["param"] for item in test_data]

    t0 = time.time()
    means, stds, valid_indices = bo.predict_batch_with_uncertainty(test_litmus, test_params)
    logger.info(f"Prediction finished in {time.time() - t0:.4f}s")

    # 结果统计
    groups = defaultdict(list)
    y_true_all = []
    y_pred_all = []

    for ptr, original_idx in enumerate(valid_indices):
        item = test_data[original_idx]
        actual = item["score"]
        pred_mean = means[ptr]
        pred_std = stds[ptr]

        groups[item["litmus"]].append({
            'param': item["param"],
            'actual': actual,
            'pred': pred_mean,
            'std': pred_std
        })
        y_true_all.append(actual)
        y_pred_all.append(pred_mean)

    # === 修改后的打印逻辑 ===
    # 列说明：
    # TOP1?       : 模型选的这个是不是第一名
    # MAX_ACT     : 这一组里，客观存在的最高分 (天花板)
    # PICK_ACT    : 模型选中的那个参数，实际跑了多少分 (我们的结果)
    # PICK_PRED   : 模型选中的那个参数，模型预测多少分 (模型的预期)
    # PICK_STD    : 模型选中的那个参数，模型的不确定性 (模型的犹豫程度)

    logger.info("=" * 100)
    logger.info(
        f"{'LITMUS':<25} | {'TOP1?':<5} | {'MAX_ACT':<7} | {'PICK_ACT':<8} | {'PICK_PRED':<9} | {'PICK_STD':<8}")
    logger.info("-" * 100)

    total_cnt = 0
    top1_cnt = 0
    top3_cnt = 0

    for litmus, records in groups.items():
        if len(records) < 2: continue
        total_cnt += 1

        # 1. 找出【真实】分数最高的记录 (客观事实)
        records_sorted_by_actual = sorted(records, key=lambda x: x['actual'], reverse=True)
        best_actual_score = records_sorted_by_actual[0]['actual']

        # 2. 找出【模型预测】分数最高的记录 (模型选择)
        records_sorted_by_pred = sorted(records, key=lambda x: x['pred'], reverse=True)
        best_pred_record = records_sorted_by_pred[0]

        # 3. 判定
        is_hit = (best_pred_record['actual'] >= best_actual_score)
        if is_hit:
            top1_cnt += 1
            match_str = "YES"
        else:
            match_str = "NO"

        # Top 3 check
        top3_preds = records_sorted_by_pred[:3]
        if any(r['actual'] >= best_actual_score for r in top3_preds):
            top3_cnt += 1

        # 打印详细信息
        # 限制打印行数或只打印错误行，防止刷屏 (可根据需要调整)
        # if total_cnt <= 20 or match_str == "NO":
        logger.info(
            f"{litmus[:25]:<25} | {match_str:<5} | {best_actual_score:<7.2f} | {best_pred_record['actual']:<8.2f} | {best_pred_record['pred']:<9.2f} | {best_pred_record['std']:<8.4f}"
        )

    # 汇总
    if total_cnt > 0:
        logger.info("=" * 60)
        logger.info(f"Top-1 Accuracy:      {top1_cnt / total_cnt * 100:.2f}%")
        logger.info(f"Top-3 Recall:        {top3_cnt / total_cnt * 100:.2f}%")

        rho, _ = spearmanr(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        logger.info(f"Global Spearman Rho: {rho:.4f}")
        logger.info(f"Global R^2 Score:    {r2:.4f}")
        logger.info("=" * 60)