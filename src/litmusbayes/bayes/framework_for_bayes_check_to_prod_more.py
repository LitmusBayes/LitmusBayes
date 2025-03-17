import json
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ================= 配置 =================
cache_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_norm_for_graph.jsonl"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
SEED = 2025


def get_interaction_features(p_vec, l_vec):
    """构造交互特征"""
    p = np.array(p_vec)
    v = np.array(l_vec)
    inter = np.outer(p, v).flatten()
    return np.concatenate([p, v, inter])


def calculate_topk_metrics(y_true, y_pred, groups, k_list=[1, 3, 5]):
    """
    计算分组 Top-K 准确率
    y_true: 真实分数数组
    y_pred: 预测分数数组
    groups: 对应的文件名数组 (用于分组)
    """
    # 1. 重新组织数据: group_data[litmus] = [(true, pred), ...]
    group_data = defaultdict(list)
    for t, p, g in zip(y_true, y_pred, groups):
        group_data[g].append((t, p))

    hits = {k: 0 for k in k_list}
    # 新增指标：Top-1 遗憾值 (Regret) - 也就是选出来的和最好的差多少
    top1_score_sum = 0
    best_score_sum = 0

    valid_groups = 0

    for g, records in group_data.items():
        # 如果测试集里这个组样本太少，比如少于5个，Top-5就没意义了，跳过
        if len(records) < 2: continue

        valid_groups += 1

        # 1. 找出【真实】最优解
        # 按真实分数排序
        records.sort(key=lambda x: x[0], reverse=True)
        actual_best_score = records[0][0]

        # 2. 找出【模型】推荐的前 K 个
        # 按预测分数排序
        records.sort(key=lambda x: x[1], reverse=True)

        if actual_best_score == 1:
            continue
        # 统计 Top-K 命中率
        for k in k_list:
            # 取模型预测的前 k 个
            candidates = records[:k]
            # 检查：这 k 个里，有没有一个人的真实分数 == 真实最优解？
            # (注意：可能有多个并列第一，只要命中其中一个就算对)
            # 为了防止浮点误差，用 >= actual - epsilon
            if any(c[0] >= actual_best_score - 1e-6 for c in candidates):
                hits[k] += 1

        # 统计 Top-1 实际得分 vs 理想得分
        model_pick_real_score = records[0][0]  # 模型排第一的那个，它的真实分数
        top1_score_sum += model_pick_real_score
        best_score_sum += actual_best_score

    # 计算平均指标
    metrics = {}
    for k in k_list:
        metrics[f"Top-{k} Acc"] = hits[k] / valid_groups

    # 归一化得分率 (1.0 代表完美，每次都选到了最好的)
    metrics["Top-1 Efficiency"] = top1_score_sum / (best_score_sum + 1e-6)

    return metrics, valid_groups


def diagnose_topk():
    print("=== 开始 Top-K 命中率实战对比 (Diagnosis) ===")

    # 1. 加载 Vector
    print("加载向量...")
    litmus_to_vec = {}
    if os.path.exists(litmus_vec_path):
        with open(litmus_vec_path, "r") as f:
            for line in f:
                if ":" in line:
                    try:
                        n, v = line.strip().split(":", 1)
                        litmus_to_vec[n] = eval(v)
                    except:
                        pass

    if not litmus_to_vec:
        print("错误：未加载到向量！")
        return

    # 2. 加载数据
    print("加载并构建特征矩阵...")
    data_objs = []

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    if obj['litmus'] in litmus_to_vec:
                        data_objs.append(obj)
                except:
                    pass

    # 构建矩阵
    X_std = []
    X_int = []
    y = []
    groups = []  # 记录每一行属于哪个文件

    for obj in data_objs:
        p = obj['param']
        l = litmus_to_vec[obj['litmus']]
        s = obj['score']

        X_std.append(list(p) + list(l))
        X_int.append(get_interaction_features(p, l))
        y.append(s)
        groups.append(obj['litmus'])

    X_std = np.array(X_std)
    X_int = np.array(X_int)
    y = np.array(y)
    groups = np.array(groups)

    # 3. 切分 (必须带上 groups 一起切)
    print(f"切分数据集 (样本数: {len(y)})...")
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=SEED)

    y_train = y[idx_train]
    y_test = y[idx_test]
    groups_test = groups[idx_test]  # 评估只看测试集

    y_train_log = np.log1p(y_train)

    # 4. 训练与预测

    # === Model A: Standard ===
    print(">>> 训练模型 A (Standard)...")
    model_std = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features= "sqrt", n_jobs=-1, random_state=SEED)
    model_std.fit(X_std[idx_train], y_train_log)
    pred_test_A = np.expm1(model_std.predict(X_std[idx_test]))

    # === Model B: Interaction ===
    print(">>> 训练模型 B (Interaction)...")
    model_int = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features= "sqrt", n_jobs=-1, random_state=SEED)
    model_int.fit(X_int[idx_train], y_train_log)
    pred_test_B = np.expm1(model_int.predict(X_int[idx_test]))

    # 5. 计算 Top-K 指标
    print("\n正在计算 Top-K 指标 (基于测试集内的排序)...")
    metrics_A, n_groups = calculate_topk_metrics(y_test, pred_test_A, groups_test)
    metrics_B, _ = calculate_topk_metrics(y_test, pred_test_B, groups_test)

    # 6. 报表展示
    print("\n" + "=" * 80)
    print(f"Top-K 准确率对比报告 (测试集覆盖文件数: {n_groups})")
    print("定义：Top-K Acc = 模型推荐的前K个参数中，是否包含该文件在测试集里的【真实第一名】")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Model A (Standard)':<20} | {'Model B (Interaction)':<20} | {'Diff'}")
    print("-" * 80)

    keys = ["Top-1 Acc", "Top-3 Acc", "Top-5 Acc", "Top-1 Efficiency"]
    for k in keys:
        val_A = metrics_A[k]
        val_B = metrics_B[k]
        diff = val_B - val_A

        # 格式化输出
        mark = ""
        if k == "Top-1 Acc" and diff > 0.01: mark = "✅ 提升"
        if k == "Top-1 Acc" and diff < -0.01: mark = "🔻 下降"

        print(f"{k:<20} | {val_A:.2%}             | {val_B:.2%}             | {diff:+.2%} {mark}")

    print("=" * 80)

    # 结论
    print("\n结论分析：")
    if metrics_B["Top-1 Acc"] > metrics_A["Top-1 Acc"]:
        print("1. 特征交叉 (Interaction) 提高了 Top-1 命中率。这说明乘法特征帮助模型更精准地锁定了峰值。")
        print("   -> 推荐使用 Interaction 模型。")
    elif metrics_B["Top-1 Acc"] < metrics_A["Top-1 Acc"]:
        print("1. 特征交叉反而降低了 Top-1 命中率。可能是特征过多导致过拟合，或者引入了噪声。")
        print("   -> 推荐使用 Standard 模型 (Param + Vec)。")
    else:
        print("1. 两者在 Top-1 上表现一致。")

    print(f"2. 当前模型的 Top-1 效率为 {metrics_A['Top-1 Efficiency']:.2%}。")
    print("   (意思是：如果模型没选到第一名，它选的那个参数的分数，平均也能达到真实最高分的百分之多少)")


if __name__ == "__main__":
    diagnose_topk()