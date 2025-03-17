import json
import logging
import os
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from collections import defaultdict

# 配置路径
cache_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_norm_for_graph.jsonl"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_dnn_gt0.log"

SEED = 2025


class EnhancedRandomForest:
    def __init__(self, litmus_names):
        # 1. 建立 ID 映射：把文件名转成 0, 1, 2... 的整数 ID
        self.litmus_id_map = {name: i for i, name in enumerate(litmus_names)}

        # 2. 加载 Vector
        self.litmus_to_vec = self._load_vec(litmus_vec_path)

        # 3. 初始化模型 (记忆模式)
        self.model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=-1,
            max_depth=None,  # 不限制深度
            min_samples_leaf=1,  # 允许死记硬背
            random_state=SEED
        )

    def _load_vec(self, path):
        d = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if ":" in line:
                        n, v = line.strip().split(":", 1)
                        d[n] = eval(v)
        return d

    def _get_features(self, litmus_name, param_vec):
        # F1: 参数 (11维)
        feats = list(param_vec)

        # F2: Vector (12维) - 即使分不开，也要加上作为辅助
        if litmus_name in self.litmus_to_vec:
            feats.extend(self.litmus_to_vec[litmus_name])
        else:
            feats.extend([0] * 12)  # 补0

        # F3: 【核心修复】Litmus ID
        # 只要加上这个，模型绝对能分清谁是谁
        lid = self.litmus_id_map.get(litmus_name, -1)
        feats.append(lid)

        return feats

    def fit_and_evaluate(self, all_data):
        print(f"正在构建特征矩阵 (样本数: {len(all_data)})...")
        X = []
        y = []

        for item in all_data:
            lname = item['litmus']
            # 过滤掉不在列表里的（如果有的话）
            if lname not in self.litmus_id_map: continue

            feats = self._get_features(lname, item['param'])
            X.append(feats)
            y.append(item['score'])

        X = np.array(X)
        y = np.array(y)

        # 切分训练/测试集 (8:2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        print(f"开始训练 (Train: {len(X_train)}, Test: {len(X_test)})...")

        # 【核心修复】Log 变换
        # 把 [0, 1800] 压缩到 [0, 7.5]，让分布更平滑
        y_train_log = np.log1p(y_train)

        self.model.fit(X_train, y_train_log)

        print("训练完成，开始评估...")

        # 预测并还原
        train_pred_log = self.model.predict(X_train)
        test_pred_log = self.model.predict(X_test)

        # 还原回真实分数
        train_pred = np.expm1(train_pred_log)
        test_pred = np.expm1(test_pred_log)

        # 计算 Rho (Ranking 能力)
        rho_train = spearmanr(y_train, train_pred).statistic
        rho_test = spearmanr(y_test, test_pred).statistic

        # 计算 R^2 (数值准确度)
        from sklearn.metrics import r2_score
        r2_train = r2_score(y_train, train_pred)
        r2_test = r2_score(y_test, test_pred)

        print("=" * 60)
        print("MODEL PERFORMANCE REPORT (Feature: Params + Vector + ID)")
        print("=" * 60)
        print(f"{'Metric':<15} | {'Train (Should be ~0.99)':<25} | {'Test (Generalization)':<20}")
        print("-" * 60)
        print(
            f"{'Spearman Rho':<15} | {rho_train:.4f} {'(PERFECT!)' if rho_train > 0.98 else '' :<20} | {rho_test:.4f}")
        print(f"{'R2 Score':<15} | {r2_train:.4f} {'(PERFECT!)' if r2_train > 0.98 else '' :<20} | {r2_test:.4f}")
        print("=" * 60)

        if rho_train > 0.95:
            print("✅ 诊断通过：模型已具备【完全记忆】能力。")
            print("   现在模型知道每个文件的脾气了，预测测试集(Test)的分数才是有意义的。")
        else:
            print("❌ 警告：模型依然无法记忆训练集。请检查特征代码是否出错。")


if __name__ == "__main__":
    # 1. 获取文件名列表
    # (为了简单，直接从 cache 文件里读所有出现过的文件名)
    print("读取数据中...")
    all_data = []
    litmus_set = set()

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        all_data.append(obj)
                        litmus_set.add(obj['litmus'])
                    except:
                        pass
    else:
        print("找不到数据文件！")
        exit()

    print(f"共加载 {len(all_data)} 条数据，涉及 {len(litmus_set)} 个文件。")

    # 2. 初始化增强版模型
    # 把 set 转 list 固定顺序
    rf = EnhancedRandomForest(list(litmus_set))

    # 3. 运行评估
    rf.fit_and_evaluate(all_data)