import json
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# ================= 配置 =================
# 保持和你刚才一样的路径
cache_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_norm_for_graph.jsonl"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
SEED = 2025


def get_interaction_features(p_vec, l_vec):
    """
    构造交互特征：[P, V, P*V]
    """
    p = np.array(p_vec)
    v = np.array(l_vec)
    # 外积展平：生成 P_i * V_j 的所有组合
    inter = np.outer(p, v).flatten()
    return np.concatenate([p, v, inter])


def diagnose_interaction():
    print("=== 开始特征交互效果验证 (Diagnosis) ===")

    # 1. 加载 Vector
    print("加载向量表...")
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

    # 自动检测向量维度
    if litmus_to_vec:
        vec_dim = len(next(iter(litmus_to_vec.values())))
        print(f"检测到向量维度: {vec_dim}")
    else:
        print("错误：未加载到向量！")
        return

    # 2. 加载数据
    data = []

    print("加载训练数据...")
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    if obj['litmus'] in litmus_to_vec:
                        data.append(obj)
                except:
                    pass
    else:
        print("错误：找不到缓存文件")
        return

    print(f"有效样本数: {len(data)}")

    # 3. 准备两组特征矩阵
    print("\n正在构建特征矩阵...")

    X_standard = []  # 旧方案: [Param, Vec]
    X_interaction = []  # 新方案: [Param, Vec, Param*Vec]
    y = []

    for obj in data:
        p_vec = obj['param']
        l_vec = litmus_to_vec[obj['litmus']]
        score = obj['score']

        # 旧特征
        X_standard.append(list(p_vec) + list(l_vec))

        # 新特征
        X_interaction.append(get_interaction_features(p_vec, l_vec))

        y.append(score)

    X_standard = np.array(X_standard)
    X_interaction = np.array(X_interaction)
    y = np.array(y)

    print(f"基准特征维度: {X_standard.shape[1]}")
    print(f"交互特征维度: {X_interaction.shape[1]}")

    # 4. 训练对比
    # 切分数据 (保证两组模型用同样的切分)
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=SEED)

    y_train = y[idx_train]
    y_test = y[idx_test]

    # 为了公平，都使用 Log 变换
    y_train_log = np.log1p(y_train)

    # --- 模型 A: 基准 (Standard) ---
    print("\n>>> 训练模型 A (Standard: Param + Vec)...")
    model_std = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=1, max_depth=None, n_jobs=-1, random_state=SEED
    )
    model_std.fit(X_standard[idx_train], y_train_log)

    # 预测并还原
    pred_train_A = np.expm1(model_std.predict(X_standard[idx_train]))
    pred_test_A = np.expm1(model_std.predict(X_standard[idx_test]))

    rho_train_A = spearmanr(y_train, pred_train_A).statistic
    rho_test_A = spearmanr(y_test, pred_test_A).statistic

    # --- 模型 B: 交互 (Interaction) ---
    print("\n>>> 训练模型 B (Enhanced: Param * Vec)...")
    model_int = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=1, max_depth=None, n_jobs=-1, random_state=SEED
    )
    model_int.fit(X_interaction[idx_train], y_train_log)

    pred_train_B = np.expm1(model_int.predict(X_interaction[idx_train]))
    pred_test_B = np.expm1(model_int.predict(X_interaction[idx_test]))

    rho_train_B = spearmanr(y_train, pred_train_B).statistic
    rho_test_B = spearmanr(y_test, pred_test_B).statistic

    # 5. 结果展示
    print("\n" + "=" * 70)
    print(f"{'Metric':<20} | {'Model A (Standard)':<20} | {'Model B (Interaction)':<20}")
    print("-" * 70)

    diff_train = rho_train_B - rho_train_A
    diff_test = rho_test_B - rho_test_A

    print(f"{'Train Rho':<20} | {rho_train_A:.4f} {'(Base)':<14} | {rho_train_B:.4f} ({diff_train:+.4f})")
    print(f"{'Test Rho':<20} | {rho_test_A:.4f} {'(Base)':<14} | {rho_test_B:.4f} ({diff_test:+.4f})")
    print("=" * 70)

    # 结论判定
    if rho_train_B > 0.85:
        print("\n✅ 诊断成功：特征交叉显著提升了模型的区分能力！")
        print("   Train Rho 达到了较高水平，说明 [参数*向量] 能够让模型区分不同的 Litmus Test。")
        print("   -> 建议：可以直接使用 Interaction 模式进行参数推荐。")
    elif rho_train_B > rho_train_A + 0.1:
        print("\n⚠️ 诊断结果：有提升，但未达完美。")
        print("   特征交叉确实有效 (Train Rho 提升明显)，但可能还需要更多数据或更强的向量。")
        print("   -> 建议：可以使用 Interaction 模式，效果会比之前好。")
    else:
        print("\n❌ 诊断失败：特征交叉效果不明显。")
        print("   这说明你的 Litmus Vector 可能丢失了太多关键信息，乘法也救不回来。")
        print("   -> 建议：如果不能用 One-Hot，可能需要重新设计 Litmus Vector。")


if __name__ == "__main__":
    diagnose_interaction()