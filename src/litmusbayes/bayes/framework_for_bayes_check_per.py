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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.litmusbayes.bayes.logger_util import setup_logger, get_logger

SEED = 2025
LOG_NAME = "bayes_eval"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"


def evaluate_per_task_model():
    # 1. Setup
    random.seed(SEED)
    np.random.seed(SEED)
    log_file_name = f"{stat_log_base}.per_task.run.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Per-Task Evaluation | Seed={SEED} ===")

    # 2. 加载数据
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
        exit(1)

    # 3. 按 Litmus Test 分组
    groups = defaultdict(list)
    for item in all_data:
        groups[item['litmus']].append(item)

    logger.info(f"Total Groups: {len(groups)}")

    # 4. 开始逐个训练和评估
    global_results = []

    # 统计计数器
    total_samples = 0
    top1_hit = 0
    top3_hit = 0

    processed_groups = 0

    for litmus_name, records in groups.items():
        # === 关键限制：样本量 ===
        # 如果样本太少（比如少于20个），Random Forest 根本跑不起来，或者全是过拟合
        if len(records) < 20:
            continue

        processed_groups += 1

        # 提取特征 (只用 Param!) 和 标签
        X = np.array([r['param'] for r in records])
        # 使用 log1p 平滑
        y = np.log1p(np.array([r['score'] for r in records]))
        y_raw = np.array([r['score'] for r in records])

        # 即使是单模型，也要切分训练集和测试集来验证泛化能力
        # 训练集 70% (约 40-100个样本)，测试集 30% (约 20-40个样本)
        try:
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, np.arange(len(records)), test_size=0.3, random_state=SEED
            )
        except ValueError:
            continue

        # === 训练专属模型 ===
        # 因为样本极少，我们要限制树的深度，防止死记硬背
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,  # 限制深度，防止过拟合
            min_samples_leaf=2,
            n_jobs=1,  # 单个模型用单核，因为外层可以并行(虽然这里是串行循环)
            random_state=SEED
        )

        model.fit(X_train, y_train)

        # === 预测 ===
        pred_test_log = model.predict(X_test)
        pred_test = np.expm1(pred_test_log)
        y_test_raw = np.expm1(y_test)

        # === 评估 Ranking ===
        # 我们只关心测试集里的排序
        test_records = []
        for i in range(len(y_test)):
            test_records.append({
                'actual': y_test_raw[i],
                'pred': pred_test[i]
            })

        # 找真实最优
        test_records.sort(key=lambda x: x['actual'], reverse=True)
        max_actual = test_records[0]['actual']

        # 找预测最优
        pred_sorted = sorted(test_records, key=lambda x: x['pred'], reverse=True)

        # Top-1 Check
        if pred_sorted[0]['actual'] >= max_actual:
            top1_hit += 1

        # Top-3 Check
        if any(r['actual'] >= max_actual for r in pred_sorted[:3]):
            top3_hit += 1

        total_samples += 1

        # 打印部分日志
        if processed_groups % 50 == 0:
            logger.info(f"Processed {processed_groups} groups...")

    # === 最终汇总 ===
    logger.info("=" * 60)
    logger.info("      PER-TASK MODEL (ONE RF PER LITMUS)      ")
    logger.info("=" * 60)
    logger.info(f"Valid Groups (Samples > 20): {total_samples}")

    if total_samples > 0:
        logger.info(f"Top-1 Accuracy: {top1_hit / total_samples * 100:.2f}%")
        logger.info(f"Top-3 Recall:   {top3_hit / total_samples * 100:.2f}%")
    else:
        logger.warning("No valid groups found!")

    logger.info("=" * 60)


if __name__ == "__main__":
    evaluate_per_task_model()