import json
import statistics
import os
import shutil


# ============================
# 1. 数据加载模块 (已修改: 支持混合格式)
# ============================
regressed_tests = []
with open("../RQ2/regressed_litmus.txt", 'r') as f:
    for line in f:
        name = line.strip()
        if name:  # 忽略空行
            regressed_tests.append(name)

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
        if name in regressed_tests:
            with open("regressed_score.txt", 'a+') as f:
                f.write(f"{name},{score_base}|{score_new}|{score_new/score_base}\n")
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

    # 4. 打印性能提升的
    improved = results['improved']
    if improved:
        improved_sorted = sorted(improved, key=lambda x: x['ratio'], reverse=True)
        print("\n" + "-" * 80)
        print(f"🟢 性能提升列表 (共 {len(improved)} 个)")
        print("-" * 80)
        print(f"{'Test Name':<50} | {'Old Score':<12} | {'New Score':<12} | {'Ratio':<8}")
        for item in improved_sorted[:10]:  # 只打印前10个避免刷屏
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


# ============================
# 6. 执行入口
# ============================

# !!! 请在此处修改你的 Log 路径 !!!
log_path1 = './baseline_scores.json'
log_path2 = './log_record_best.log.validation_cache.jsonl'
# log_path1 = './log_record_init.log.validation_cache.jsonl'

# !!! 请在此处修改你的文件夹路径 !!!
LITMUS_SOURCE_DIR = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive'
STATUS_CHANGE_DEST_DIR = './status_diff_litmus'



try:
    with open(log_path1, 'r', encoding='utf-8') as f:
        log1_content = f.read()
    with open(log_path2, 'r', encoding='utf-8') as f:
        log2_content = f.read()

    data1 = load_log1_data(log1_content)
    data2 = load_log_data(log2_content)

    # 1. 分析
    analysis_results = analyze_comparison(data1, data2)

    # 2. 打印详细列表 (包含状态变化的)
    print_detailed_changes(analysis_results)

    # 3. 打印时间对比
    compare_both_passed_time(data1, data2)

    # 4. 打印整体性能统计与新增测试信息 (新增调用)
    print_summary_statistics(analysis_results)

    # 5. 自动提取状态改变的文件 (按需解除注释)
    # copy_status_changed_files(
    #     analysis_results,
    #     LITMUS_SOURCE_DIR,
    #     STATUS_CHANGE_DEST_DIR
    # )

except FileNotFoundError as e:
    print(f"错误: 找不到文件 - {e}")