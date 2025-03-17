import json
import os

# 配置输入和输出文件名
INPUT_FILE = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_representative_run.log.cache.jsonl'  # 请将你的源文件名放在这里
OUTPUT_FILE = '/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_representative_norm_run.log.cache.jsonl'

# 定义基准 param
BASELINE_PARAM = [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0]


def normalize_json_scores():
    # 1. 读取数据并按 litmus 名称分组
    grouped_data = {}

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                try:
                    record = json.loads(line)
                    litmus_name = record.get('litmus')

                    if litmus_name not in grouped_data:
                        grouped_data[litmus_name] = []

                    grouped_data[litmus_name].append(record)
                except json.JSONDecodeError:
                    print(f"跳过无效的 JSON 行: {line}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    normalized_results = []

    # 2. 处理每一组数据
    for litmus, records in grouped_data.items():
        baseline_score = None

        # 寻找基准分
        for r in records:
            if r.get('param') == BASELINE_PARAM:
                baseline_score = r.get('score')
                break

        # 3. 检查基准分是否有效 (存在且不为0)
        # 注意：如果不包含基准 param，或者基准 param 的分为 0，则跳过整组 (即删除)
        if baseline_score is not None and baseline_score != 0:
            for r in records:
                # 复制对象以避免修改原始引用（虽然这里不重要，但好习惯）
                new_record = r.copy()

                # 执行归一化：当前分数 / 基准分数
                original_score = r.get('score', 0)
                new_record['score'] = original_score / baseline_score

                normalized_results.append(new_record)
        else:
            status = "分数为0" if baseline_score == 0 else "未找到基准param"
            print(f"删除 Litmus 测试组: {litmus} (原因: {status})")

    # 4. 写入新文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in normalized_results:
            f.write(json.dumps(record) + '\n')

    print(f"处理完成。结果已保存至 {OUTPUT_FILE}")
    print(f"原始记录数: {sum(len(v) for v in grouped_data.values())}")
    print(f"保留记录数: {len(normalized_results)}")


if __name__ == "__main__":
    normalize_json_scores()