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
cache_file_path = stat_log_base + ".cache4_norm.jsonl"

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
    # 1. 生成全量参数候选集 (约 20,160 个组合)
    logger.info("Generating all possible parameter combinations...")
    all_candidates = param_space.get_all_combinations()
    logger.info(f"Total candidate combinations: {len(all_candidates)}")

    # 2. 准备结果容器
    # 格式: { "litmus_name": { "best_param": [...], "pred_score": 0.95 }, ... }
    optimization_results = {}

    # 3. 遍历所有 Litmus 文件进行预测
    # 这里的 litmus_names 是你之前加载的所有测试用例列表
    # 如果数量太多(比如几千个)，这一步可能需要几分钟

    # 为了提升速度，我们将候选集转为 numpy array 一次，避免循环中重复转换
    import numpy as np

    # X_base_params: shape (N_candidates, 11)
    X_base_params = np.array(all_candidates)

    count = 0
    total_tasks = len(litmus_names)

    for litmus in litmus_names:
        count += 1
        if count % 100 == 0:
            logger.info(f"Processing {count}/{total_tasks}...")

        # 获取该 Litmus 对应的静态特征向量 (Litmus Vector)
        if litmus not in bo.litmus_to_vector_dict:
            logger.warning(f"Missing vector for {litmus}, skipping.")
            continue

        litmus_vec = bo.litmus_to_vector_dict[litmus]
        # litmus_vec shape: (dim_vec, )

        # --- 批量构造输入矩阵 ---
        # 我们需要将 (1, dim_vec) 广播复制到 (N_candidates, dim_vec)
        # 然后和 X_base_params 拼接

        # 方法：利用 numpy 的广播机制或 tile
        # 这里的 all_candidates 有 2万行，直接 python 循环拼 list 会慢，用 numpy 操作

        # 构造 Litmus 特征矩阵: 形状 (N_candidates, len(litmus_vec))
        # np.tile 会把向量重复 N 次
        X_litmus_part = np.tile(litmus_vec, (len(X_base_params), 1))

        # 拼接: [Params | LitmusVec]
        # axis=1 表示横向拼接
        X_batch = np.hstack([X_base_params, X_litmus_part])

        # --- 批量预测 ---
        # 模型输出的是 log1p 后的值
        pred_logs = bo.model.predict(X_batch)

        # --- 找最大值 ---
        best_idx = np.argmax(pred_logs)
        max_log_val = pred_logs[best_idx]

        # 还原分数
        best_score = np.expm1(max_log_val)
        best_param_vec = all_candidates[best_idx]  # 从原始列表中取，保证格式纯净

        # 记录结果
        optimization_results[litmus] = {
            "param": best_param_vec,
            "pred_score": float(best_score)  # 转成 float 以便 json 序列化
        }

    # 4. 输出或保存结果
    logger.info("=" * 60)
    logger.info(f"Optimization Completed for {len(optimization_results)} files.")

    # 保存到 JSON
    output_file = "best_params_recommendation_final.json"
    with open(output_file, "w") as f:
        json.dump(optimization_results, f, indent=4)

    logger.info(f"Results saved to {output_file}")

    # 5. (可选) 打印几个示例看看
    logger.info("Sample Recommendations:")
    sample_keys = list(optimization_results.keys())[:5]
    for k in sample_keys:
        res = optimization_results[k]
        # 把 param vector 转回人类可读的字典打印出来，方便检查
        readable_param = param_space.vector_to_param_dict(res['param'])
        logger.info(f"File: {k}")
        logger.info(f"  -> Score: {res['pred_score']:.4f}")
        logger.info(f"  -> Param: {readable_param}")
