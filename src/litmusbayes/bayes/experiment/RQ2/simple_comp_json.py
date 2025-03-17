import json
import statistics
import os
import shutil


# ============================
# 1. 数据加载模块 (已修改: 支持混合格式)
# ============================

def load_log1_data(content):
    """
    专门用于解析 Log1 新格式："Key": Value,
    支持带有大括号的标准 JSON 字典，也支持逐行键值对。
    """
    data = {}

    # 情况A：尝试直接整体解析（以防文件其实有完整 {} 包裹）
    try:
        full_json = json.loads(content)
        if isinstance(full_json, dict):
            for name, raw_score in full_json.items():
                score = float(raw_score)
                data[name] = max(data.get(name, score), score)
            return data
    except json.JSONDecodeError:
        pass

    # 情况B：按行解析纯键值对 (应对没有外层 {} 或直接粘贴的文本)
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line in ('{', '}'):
            continue

        # 去掉末尾的逗号
        if line.endswith(','):
            line = line[:-1]

        try:
            # 巧妙包装上一层大括号，使其成为合法的 JSON 行
            entry = json.loads('{' + line + '}')
            for name, raw_score in entry.items():
                score = float(raw_score)
                if name in data:
                    data[name] = max(data[name], score)
                else:
                    data[name] = score
        except Exception:
            continue

    return data


def load_log_data(content):
    """
    原有的解析逻辑，继续用于 Log2 的 JSONL 格式解析。
    格式要求: {"litmus": "xxx", "score": yyy}
    """
    data = {}
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        try:
            entry = json.loads(line)
            name = entry.get("litmus")
            raw_score = entry.get("score")
            if name is None or raw_score is None: continue
            score = float(raw_score)
            if name in data:
                data[name] = max(data[name], score)
            else:
                data[name] = score
        except json.JSONDecodeError:
            continue
    return data


# ============================
# 2. 核心分析模块 (已修改)
# ============================

def analyze_comparison(data_base, data_new):
    time1 = 0
    time2 = 0
    results = {
        "improved": [], "regressed": [], "new_pass": [],
        "new_fail": [], "errors": [], "same": 0, "ratios": [],
        "total_compared": 0, "completely_new_pass": []  # 新增 completely_new_pass 记录
    }

    for name, score_new in data_new.items():
        # 处理 Log1 中没有的 Litmus Test
        if name not in data_base:
            if score_new > 0:
                results["completely_new_pass"].append({"name": name, "new": score_new})
            continue

        score_base = data_base[name]
        results["total_compared"] += 1

        if score_base > 0 and score_new > 0:
            diff = score_new - score_base
            ratio = score_new / score_base
            item = {"name": name, "base": score_base, "new": score_new, "diff": diff, "ratio": ratio}

            if score_new > score_base:
                results["improved"].append(item)
                results["ratios"].append(ratio)
                time1 += 3 / score_new
                time2 += 3 / score_base
            elif score_new < score_base:
                results["regressed"].append(item)
                results["ratios"].append(ratio)
            else:
                results["same"] += 1
                results["ratios"].append(1.0)
        # Log1 <= 0, Log2 > 0 (复活/修复 - 原本在Log1中存在但失败)
        elif score_base <= 0 and score_new > 0:
            results["new_pass"].append({"name": name, "base": score_base, "new": score_new})
        # Log1 > 0, Log2 <= 0 (挂掉/失效)
        elif score_base > 0 and score_new <= 0:
            results["new_fail"].append({"name": name, "base": score_base, "new": score_new})
            if score_new == -1: results["errors"].append(name)
        else:
            results["same"] += 1
            if score_new == -1: results["errors"].append(name)

    return results


# ============================
# 3. 详细输出模块
# ============================

def print_detailed_changes(results):
    # 1. 打印从 Pass 变成 Fail 的 (Log1 > 0, Log2 <= 0)
    new_fails = results['new_fail']
    if new_fails:
        print("\n" + "!" * 80)
        print(f"❌ 新增失败列表 (Pass -> Fail) (共 {len(new_fails)} 个)")
        print("!" * 80)
        print(f"{'Test Name':<50} | {'Old Score':<12} | {'New Score':<12}")
        print("-" * 80)
        for item in new_fails:
            print(f"{item['name']:<50} | {item['base']:<12.2f} | {item['new']:<12.2f}")

    # 2. 打印从 Fail 变成 Pass 的 (Log1 <= 0, Log2 > 0)
    new_passes = results['new_pass']
    if new_passes:
        print("\n" + "*" * 80)
        print(f"✅ 修复通过列表 (Fail -> Pass) (共 {len(new_passes)} 个)")
        print("*" * 80)
        print(f"{'Test Name':<50} | {'Old Score':<12} | {'New Score':<12}")
        print("-" * 80)
        for item in new_passes:
            print(f"{item['name']:<50} | {item['base']:<12.2f} | {item['new']:<12.2f}")

    # 3. 打印性能下降的
    regressed = results['regressed']
    if regressed:
        regressed_sorted = sorted(regressed, key=lambda x: x['ratio'])
        print("\n" + "-" * 80)
        print(f"🔻 性能下降列表 (共 {len(regressed)} 个) - 按下降幅度排序")
        print("-" * 80)
        print(f"{'Test Name':<50} | {'Old Score':<12} | {'New Score':<12} | {'Ratio':<8}")
        for item in regressed_sorted:
            print(f"{item['name']:<50} | {item['base']:<12.2f} | {item['new']:<12.2f} | x{item['ratio']:.2f}")

        with open("regressed_litmus.txt", 'w') as f:
            for item in regressed_sorted:
                f.write(item['name'] + '\n')


    # 4. 打印性能提升的
    improved = results['improved']
    if improved:
        improved_sorted = sorted(improved, key=lambda x: x['ratio'], reverse=True)
        print("\n" + "-" * 80)
        print(f"🟢 性能提升列表 (共 {len(improved)} 个)")
        print("-" * 80)
        print(f"{'Test Name':<50} | {'Old Score':<12} | {'New Score':<12} | {'Ratio':<8}")
        for item in improved_sorted[:]:  # 只打印前10个避免刷屏
            print(f"{item['name']:<50} | {item['base']:<12.2f} | {item['new']:<12.2f} | x{item['ratio']:.2f}")
        if len(improved) > 10: print(f"... 以及其他 {len(improved) - 10} 个")


# ============================
# 4. 统计与时间模块 (新增统计打印)
# ============================

def calculate_time_for_subset(data_map, subset_keys):
    total = 0.0
    for key in subset_keys:
        if data_map[key] > 0:
            total += (3.0 / data_map[key])
    return total



def compare_both_passed_time(data_base, data_new):
    passed_keys = [k for k in data_base.keys() if k in data_new and data_base[k] > 0 and data_new[k] > 0]
    # print(passed_keys)
    remove_keys = [
                   # "MP+fence.rw.rw+ctrl-cleaninit",
                   # "MP+fence.rw.rw+ctrl",
                   # "SB+po-ctrlfencei+pos-po-addr",
                   # "MP+fence.rw.w+ctrl",
                   # "SB+po-addr+po-ctrlfencei",
                   # "SB+pos-po+pos-po-ctrlfencei",
                   # "SB+po+pos-po-ctrlfencei",
                   # "MP+fence.w.w+ctrl",
                   # "SB+pos-addr+pos-ctrlfencei",
                   # "SB+pos-addr+pos-pos-ctrlfencei",
                   # "SB+pos-po+po-ctrlfencei",
                   # "SB+pos-ctrlfencei+pos-pos-addr",
                   # "SB+po+po-ctrlfencei",
                   # "SB+po-addrs+po-ctrlfencei",
                   # "SB+pos-po-addrs+pos-po-ctrlfencei",
                   # "MP+fence.rw.w+po",
                   # "SB+rfi-ctrlfencei+pos-rfi-addr",
                   # "SB+po+pos-po-ctrlfenceis",
                   # "SB+po+pos-po",
                   # "R+rfi-ctrlfencei+rfi-addr",
                   # "SB+pos-po-addr+pos-po-ctrlfencei",
                   # "MP+poprl+po+NEW",
                   # "SB+pos-po-addr+pos-po-ctrlfenceis",
                   # "MP+fence.rw.rw+po",
                   # "SB+po-ctrlfencei+pos-po-addrs",
                   # "SB+po-addrs+pos-po-ctrlfenceis",
                   # "MP+wsi-rfi-data+ctrl",
                   # "SB+po-addrs+po-ctrlfenceis",
                   # "SB+po-addr+pos-po-ctrlfencei",
                   # "MP+wsi-rfi-addr+ctrl",
                   # "MP+wsi-rfi-ctrl+ctrl",
                   # "MP+fence.w.w+po",
                   # "SB+po-addr+po-ctrlfenceis",
                   ]
    for key in remove_keys:
        passed_keys.remove(key)
    t1 = calculate_time_for_subset(data_base, passed_keys)
    t2 = calculate_time_for_subset(data_new, passed_keys)

    print("\n" + "=" * 60)
    print(f"🚀 纯净性能对比 (仅 Both Passed)")
    print("=" * 60)
    if t1 > 0:
        reduction = (t1 - t2) / t1 * 100
        print(f"Log1 总耗时: {t1:.4f}s")
        print(f"Log2 总耗时: {t2:.4f}s")
        print(f"结论: {'变快' if t2 < t1 else '变慢'} {abs(reduction):.2f}%")

def only_compare_both_faster_time(data_base, data_new):
    passed_keys = [k for k in data_base.keys() if k in data_new and data_base[k] > 0 and data_new[k] > 0 and data_new[k] > data_base[k]]
    t1 = calculate_time_for_subset(data_base, passed_keys)
    t2 = calculate_time_for_subset(data_new, passed_keys)

    print("\n" + "=" * 60)
    print(f"🚀 faster性能对比 (仅 Both Passed and log2 faster)")
    print("=" * 60)
    if t1 > 0:
        reduction = (t1 - t2) / t1 * 100
        print(f"Log1 总耗时: {t1:.4f}s")
        print(f"Log2 总耗时: {t2:.4f}s")
        print(f"结论: {'变快' if t2 < t1 else '变慢'} {abs(reduction):.2f}%")

def compare_both_time_union(data_base, data_new):
    passed_keys = [k for k in data_base.keys() if k in data_new and data_base[k] > 0 and data_new[k] > 0]
    passed_keys_t1 = [k for k in data_base.keys() if k in data_new and data_base[k] > 0 and data_new[k] != -1]
    passed_keys_t2 = [k for k in data_new.keys() if k in data_new and data_new[k] > 0]
    errored_keys = [k for k in data_base.keys() if k in data_new and data_new[k] == -1 and data_base[k] >0]

    prod_score = 3

    t1 = calculate_time_for_subset(data_base, passed_keys_t1) + prod_score * (len(passed_keys_t2) - len(passed_keys))
    t2 = calculate_time_for_subset(data_new, passed_keys_t2) + prod_score * (len(passed_keys_t1) - len(passed_keys))
    print (len(passed_keys_t2) - len(passed_keys))
    print (len(passed_keys_t1))
    print("\n" + "=" * 60)
    print(f"🚀 union性能对比 (1s)")
    print("=" * 60)
    if t1 > 0:
        reduction = (t1 - t2) / t1 * 100
        print(f"Log1 总耗时: {t1:.4f}s")
        print(f"Log2 总耗时: {t2:.4f}s")
        print(f"结论: {'变快' if t2 < t1 else '变慢'} {abs(reduction):.2f}%")

def print_summary_statistics(results):
    """打印新增的统计指标：比例、平均值、中位数、新增通过数"""
    print("\n" + "=" * 60)
    print("📊 总体性能提升与新增统计")
    print("=" * 60)

    improved_count = len(results.get('improved', []))
    regressed_count = len(results.get('regressed', []))
    same_count = results.get('same', 0)

    # 比例的分母设定为：在 Log1 和 Log2 中都有分数且 >0 的测试总数
    total_valid_compared = improved_count + regressed_count + same_count

    if total_valid_compared > 0:
        improved_percentage = (improved_count / total_valid_compared) * 100
    else:
        improved_percentage = 0.0

    print(f"✅ 性能获得提升的测试比例: {improved_percentage:.2f}% ({improved_count}/{total_valid_compared})")

    improved_ratios = [item['ratio'] for item in results.get('improved', [])]
    if improved_ratios:
        mean_val = statistics.mean(improved_ratios)
        median_val = statistics.median(improved_ratios)
        print(f"🚀 提升幅度 (加速比) - 平均值: {mean_val:.2f}x, 中位数: {median_val:.2f}x")
    else:
        print("🚀 提升幅度 (加速比) - 无测试获得提升")

    completely_new_pass = len(results.get('completely_new_pass', []))
    print(f"✨ 新增通过测试 (Log1中无, 但Log2中>0): {completely_new_pass} 个")
    print("=" * 60 + "\n")


# ============================
# 5. 文件提取模块
# ============================

def _copy_files_generic(item_list, source_root, dest_root, label, file_extension=".litmus"):
    """内部通用复制函数"""
    if not item_list:
        return

    print(f"\n📂 [{label}] 开始提取文件...")
    print(f"   目标: {dest_root}")

    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    os.makedirs(dest_root, exist_ok=True)

    count = 0
    for item in item_list:
        test_name = item['name']
        filename = test_name
        # 补全后缀逻辑
        if '.' not in filename and file_extension:
            filename += file_extension

        src_path = os.path.join(source_root, f'{filename}.litmus')

        dst_path = os.path.join(dest_root, os.path.basename(filename) + ".litmus")

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            count += 1
        else:
            # 尝试不带 .litmus 后缀找一次 (容错)
            src_path_no_ext = os.path.join(source_root, filename)
            if os.path.exists(src_path_no_ext):
                shutil.copy2(src_path_no_ext, dst_path)
                count += 1
            else:
                print(f"   ⚠️ 未找到: {test_name}")

    print(f"   完成: 已复制 {count}/{len(item_list)} 个文件。")


def copy_status_changed_files(results, source_root, base_dest_dir):
    """
    提取状态改变的文件：
    """
    _copy_files_generic(
        results['new_fail'],
        source_root,
        os.path.join(base_dest_dir, "pass_to_fail"),
        "Pass -> Fail"
    )

    _copy_files_generic(
        results['new_pass'],
        source_root,
        os.path.join(base_dest_dir, "fail_to_pass"),
        "Fail -> Pass"
    )


def analyze_absolute_time_culprits(data_base, data_new, log1_total=0.8982, log2_total=23.6704):
    """
    估算每个 litmus test 的绝对耗时，并直观展示 Log2 相比 Log1 慢了多少倍
    （已适配当前 data1 和 data2 的字典格式）
    """
    tests_stats = []

    # 1. 收集有效的分数数据 (适配字典遍历)
    for name, new_score in data_new.items():
        old_score = data_base.get(name, 0.0)

        # 过滤掉得分为0或极小的异常数据，避免除以0
        if old_score > 1e-6 and new_score > 1e-6:
            tests_stats.append({
                'name': name,
                'old_score': old_score,
                'new_score': new_score
            })

    # 2. 计算权重 (耗时与 Score 成反比)
    total_inv_old = sum(1.0 / t['old_score'] for t in tests_stats)
    total_inv_new = sum(1.0 / t['new_score'] for t in tests_stats)

    results = []
    for t in tests_stats:
        # 按权重分配总时间
        time_log1 = (1.0 / t['old_score']) / total_inv_old * log1_total
        time_log2 = (1.0 / t['new_score']) / total_inv_new * log2_total

        slowdown_factor = time_log2 / time_log1 if time_log1 > 0 else float('inf')

        results.append({
            'name': t['name'],
            'time_log1': time_log1,
            'time_log2': time_log2,
            'factor': slowdown_factor,
            'new_score': t['new_score'],
            'old_score': t['old_score']
        })

    # 3. 按 Log2 中的绝对耗时从大到小排序（找出吃时间的绝对大头）
    results.sort(key=lambda x: x['time_log2'], reverse=True)

    # 4. 打印排版漂亮的输出
    print(f"\n{'=' * 100}")
    print(f"⏱️ 绝对耗时榜单 (Top 15 时间黑洞) - 按 Log2 耗时排序")
    print(f"{'=' * 100}")

    cumulative_time = 0.0
    for i, r in enumerate(results[:]):
        if r['time_log2'] < r['time_log1']:
            continue
        cumulative_time += r['time_log2']
        name_str = r['name'][:35] + ("..." if len(r['name']) > 35 else "")

        print(f"[{i + 1:02d}] {name_str:<38}")
        print(f"     => Log2 耗时: {r['time_log2']:>7.4f}s  |  Log1 耗时: {r['time_log1']:>7.4f}s  |  慢了 {r['factor']:>8.2f} 倍")

    print(f"{'-' * 100}")
    print(f"⚠️ 前 15 个测试共消耗了 Log2 约 {cumulative_time:.4f}s (占总时长 {log2_total}s 的 {cumulative_time / log2_total * 100:.2f}%)")
    print(f"{'=' * 100}\n")

    return results


# 在 __main__ 函数的最下面调用它：
# time_results = analyze_absolute_time_culprits(default_scores, final_data, log1_total=0.8982, log2_total=23.6704)

# ============================
# 6. 聚类归属分析模块 (新增)
# ============================

def load_cluster_mapping(cluster_file_path):
    """
    读取聚类 JSON 文件，建立 Test Name 到 Cluster ID 的映射
    """
    test_to_cluster = {}
    try:
        with open(cluster_file_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)

        for cluster_id, info in cluster_data.items():
            # 遍历每个聚类中的 members
            for member in info.get("members", []):
                test_to_cluster[member] = cluster_id

        print(f"✅ 成功加载聚类数据，共映射了 {len(test_to_cluster)} 个 Litmus tests。")
    except Exception as e:
        print(f"⚠️ 读取聚类文件失败: {e}")

    return test_to_cluster


def print_tests_by_cluster(test_list, test_to_cluster, category_name, icon="🔍"):
    """
    通用函数：将任意测试列表按聚类进行分组并打印
    """
    if not test_list:
        return

    cluster_to_tests = {}

    # 归类
    for item in test_list:
        test_name = item['name']
        # 如果找不到对应的聚类，就归入 "Unknown"
        cluster_id = test_to_cluster.get(test_name, "Unknown")

        if cluster_id not in cluster_to_tests:
            cluster_to_tests[cluster_id] = []
        cluster_to_tests[cluster_id].append(item)

    # 按包含的测试数量从多到少对聚类进行排序
    sorted_clusters = sorted(cluster_to_tests.items(), key=lambda x: len(x[1]), reverse=True)

    print("\n" + "=" * 80)
    print(f"{icon} {category_name} 测试的聚类分布")
    print("=" * 80)

    for cluster_id, tests in sorted_clusters:
        print(f"\n📂 聚类 ID: {cluster_id} (包含 {len(tests)} 个测试)")
        print("-" * 70)

        # 针对不同类型的列表进行格式化输出
        for t in tests:
            # 如果是 regressed/improved，带有 ratio 字段
            if 'ratio' in t:
                print(
                    f"   - {t['name']:<50} | Old: {t['base']:<6.2f} -> New: {t['new']:<6.2f} | 比例: x{t['ratio']:.2f}")
            # 如果是 new_pass/new_fail，只有 base 和 new 字段
            else:
                print(f"   - {t['name']:<50} | Old: {t['base']:<6.2f} -> New: {t['new']:<6.2f}")

# ============================
# 6. 执行入口
# ============================

# !!! 请在此处修改你的 Log 路径 !!!
# log_path1 = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes_banana.log.cache.jsonl'
log_path1 = './log_record_init_banana.log.validation_cache.jsonl'
log_path1 = './final_results/post/median_results.json'

# log_path2 = './log_record_best_banana_1000000.log.validation_cache.jsonl'
# log_path2 = './log_record_init_banana_regress.log.validation_cache.jsonl'
# log_path2 = './log_record_best_banana.log.validation_cache.jsonl'
# log_path1 = './log_record_best_post_final_kmeans_cross_15_factor_1_5_banana.log.validation_cache.jsonl'
# log_path2 = './log_record_best_post_final_debug_8_banana.log.validation_cache.jsonl'
# log_path2 = ('./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_3.log.validation_cache.jsonl')
# log_path2 = ('./log_record_best_post_final_kmeans_15_factor_1_5_banana.log.validation_cache.jsonl')
log_path2 = ('./final_results/prior/median_results.json')

# log_path1 = './log_record_init.log.validation_cache.jsonl'
CLUSTER_JSON_PATH = './cluster_results_final_10/cluster_centers.json'
# !!! 请在此处修改你的文件夹路径 !!!
LITMUS_SOURCE_DIR = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive'
STATUS_CHANGE_DEST_DIR = './status_diff_litmus'

try:
    with open(log_path1, 'r', encoding='utf-8') as f:
        log1_content = f.read()
    with open(log_path2, 'r', encoding='utf-8') as f:
        log2_content = f.read()

    data1 = load_log_data(log1_content)
    data2 = load_log_data(log2_content)

    # 1. 分析
    analysis_results = analyze_comparison(data1, data2)

    # 2. 打印详细列表 (包含状态变化的)
    print_detailed_changes(analysis_results)

    # 3. 打印时间对比
    compare_both_passed_time(data1, data2)
    only_compare_both_faster_time(data1, data2)
    compare_both_time_union(data1, data2)
    # 4. 打印整体性能统计与新增测试信息 (新增调用)
    print_summary_statistics(analysis_results)

    # 5. 自动提取状态改变的文件 (按需解除注释)
    # copy_status_changed_files(
    #     analysis_results,
    #     LITMUS_SOURCE_DIR,
    #     STATUS_CHANGE_DEST_DIR
    # )
    # === 新增的聚类分析逻辑 ===
    # === 聚类分析逻辑 ===
    # 5. 读取聚类文件并分析各个列表属于哪些聚类
    test_to_cluster_map = load_cluster_mapping(CLUSTER_JSON_PATH)

    if test_to_cluster_map:
        # 打印性能下降的聚类分布
        print_tests_by_cluster(analysis_results['regressed'], test_to_cluster_map, "性能下降 (Regressed)", "📉")

        # 打印修复通过 (Fail -> Pass) 的聚类分布
        print_tests_by_cluster(analysis_results['new_pass'], test_to_cluster_map, "修复通过 (Fail -> Pass)", "✅")

        # 你甚至还可以顺手把新增失败的也打出来看看
        print_tests_by_cluster(analysis_results['new_fail'], test_to_cluster_map, "新增失败 (Pass -> Fail)", "❌")
    # =========================

    # time_results = analyze_absolute_time_culprits(data1, data2, log1_total=0.8982, log2_total=23.6704)

except FileNotFoundError as e:
    print(f"错误: 找不到文件 - {e}")