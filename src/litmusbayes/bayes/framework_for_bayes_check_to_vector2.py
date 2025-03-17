import json
import numpy as np
import os
from collections import defaultdict
import statistics

# ================= 配置 =================
# 你的新向量文件路径
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
# 你的历史数据缓存
cache_file_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log.cache_sum_70_no_norm_for_graph.jsonl"


def analyze_param_level_consistency():
    print("=== 开始参数级深度一致性校验 (Param-Level Consistency Check) ===")

    # 1. 加载向量并分组
    print("1. 正在加载向量并聚类...")
    vec_to_files = defaultdict(list)

    if os.path.exists(litmus_vec_path):
        with open(litmus_vec_path, "r") as f:
            for line in f:
                if ":" in line:
                    try:
                        n, v = line.strip().split(":", 1)
                        vec_tuple = tuple(eval(v))
                        vec_to_files[vec_tuple].append(n)
                    except:
                        pass
    else:
        print("错误：找不到向量文件")
        return

    print(f"   唯一向量数: {len(vec_to_files)}")

    # 2. 加载详细历史数据 (File -> Param -> Score)
    print("2. 正在加载详细历史数据...")
    # 结构: file_data[filename][param_tuple] = score
    file_data = defaultdict(dict)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    lname = obj['litmus']
                    param = tuple(obj['param'])  # 转 tuple
                    score = obj['score']

                    # 如果同一个文件同一个参数跑了多次，取最新或平均？这里取最后一次覆盖
                    file_data[lname][param] = score
                except:
                    pass

    # 3. 开始逐个参数对比
    print("3. 开始对比同向量、同参数下的分数差异...")

    # 统计数据容器
    all_diffs_abs = []  # 绝对误差
    all_diffs_rel = []  # 相对误差
    conflict_cases = []  # 严重冲突案例

    # 阈值：如果在同一参数下，分数差异超过 30%，且绝对分差大于 0.5 (过滤掉0 vs 0.1这种)，算冲突
    REL_THRESHOLD = 0.3
    ABS_THRESHOLD = 0.5

    checked_groups = 0
    checked_params = 0

    for vec, files in vec_to_files.items():
        # 必须至少有2个文件才能对比
        if len(files) < 2: continue

        # 这一组文件里，收集所有跑过的参数集合
        # 我们只关心“交集”或者“至少被2个文件跑过”的参数
        # 为了高效，我们统计每个参数被该组内几个文件跑过
        param_counter = defaultdict(list)  # param -> [ (filename, score), ... ]

        for fname in files:
            if fname not in file_data: continue
            for p, s in file_data[fname].items():
                param_counter[p].append((fname, s))

        # 检查该组内的对比情况
        group_has_conflict = False

        for p, run_list in param_counter.items():
            # 只有当这个参数被 >= 2 个文件跑过，才有对比意义
            if len(run_list) < 2: continue

            checked_params += 1
            scores = [x[1] for x in run_list]

            max_s = max(scores)
            min_s = min(scores)
            mean_s = np.mean(scores)

            diff_abs = max_s - min_s

            # 计算相对误差
            if max_s < 1e-6:
                diff_rel = 0.0
            else:
                diff_rel = diff_abs / max_s

            all_diffs_abs.append(diff_abs)
            all_diffs_rel.append(diff_rel)

            # 判定冲突
            if diff_rel > REL_THRESHOLD and diff_abs > ABS_THRESHOLD:
                # 记录最严重的几个
                conflict_cases.append({
                    "vec": vec,
                    "param": p,
                    "scores": run_list,  # [(file, score)...]
                    "diff_rel": diff_rel,
                    "diff_abs": diff_abs
                })

        checked_groups += 1

    # 4. 输出报告
    print("\n" + "=" * 60)
    print("深度一致性分析报告")
    print("=" * 60)
    print(f"检查的 Vector 组数: {checked_groups}")
    print(f"检查的 (Vector, Param) 组合对数: {checked_params}")

    if checked_params == 0:
        print("警告：没有找到任何重叠的参数运行记录。")
        print("原因可能是：虽然文件共用向量，但它们各自跑的参数完全没有交集。")
        return

    avg_abs_diff = np.mean(all_diffs_abs)
    avg_rel_diff = np.mean(all_diffs_rel)
    median_rel_diff = np.median(all_diffs_rel)

    print(f"\n平均绝对分差 (MAE): {avg_abs_diff:.4f}")
    print(f"平均相对误差 (MRE): {avg_rel_diff:.2%}")
    print(f"中位数相对误差:     {median_rel_diff:.2%}")

    print("-" * 60)
    print(f"严重冲突点 (差异 > {REL_THRESHOLD:.0%}) 数量: {len(conflict_cases)} / {checked_params}")

    # 排序并打印 Top 10 冲突
    # 按相对误差排序
    conflict_cases.sort(key=lambda x: x['diff_abs'], reverse=True)

    print("\n[Top 10 最离谱的冲突案例]")
    print("(如果这里也是空白，或者分差很小，说明向量完全可用！)")
    for i in range(min(10, len(conflict_cases))):
        c = conflict_cases[i]
        print(f"\nCase #{i + 1}: 差异 {c['diff_rel']:.1%} (Abs: {c['diff_abs']:.2f})")
        # print(f"  Vector: {c['vec'][:3]}...")
        print(f"  Param:  {c['param']}")
        print(f"  Details:")
        for fname, score in c['scores']:
            print(f"    - {fname:<40} -> Score: {score:.2f}")

    print("\n" + "=" * 60)
    print("结论建议：")
    if avg_abs_diff < 1.0 and median_rel_diff < 0.1:
        print("✅ 完美！同向量下，参数跑分差异极小。")
        print("   这意味着'重复向量'仅仅是同一个逻辑文件的不同副本，或者运行噪声。")
        print("   -> 这种向量是非常高质量的，请放心使用。")
    elif len(conflict_cases) > checked_params * 0.1:
        print("❌ 危险！存在大量冲突。")
        print("   同一种向量，同一个参数，居然跑分天差地别。")
        print("   -> 向量分辨率不足，请换 TF-IDF。")
    else:
        print("⚠️ 中规中矩。存在一些噪声，但在可控范围内。")
        print("   -> 建议结合特征交叉 (Interaction) 使用。")


if __name__ == "__main__":
    analyze_param_level_consistency()