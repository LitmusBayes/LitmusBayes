import json


def filter_litmus_sequences(blacklist_path, sequence_path, output_path):
    # 1. 提取黑名单中的名字
    blacklist = set()
    with open(blacklist_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                # 提取 "|" 之前的部分并去除多余空格
                name = line.split('|')[0].strip()
                if name:
                    blacklist.add(name)

    print(f"成功加载黑名单，共 {len(blacklist)} 个项。")

    # 2. 处理第二个文件并过滤
    count = 0
    removed_count = 0

    with open(sequence_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            # 假设文件2的格式是 "Name: Sequence"
            if ':' in line:
                name = line.split(':', 1)[0].strip()

                if name in blacklist:
                    removed_count += 1
                    continue

                fout.write(line + '\n')
                count += 1
            else:
                # 如果某行不含冒号，默认保留或根据需要处理
                fout.write(line + '\n')

    print(f"处理完成！")
    print(f"过滤掉的项数: {removed_count}")
    print(f"保留的项数: {count}")
    print(f"结果已保存至: {output_path}")
# --- 配置路径 ---
blacklist_file = '/home/whq/Desktop/code_list/perple_test/bayes_stat/filtered_litmus_log.txt'  # 你的第一个文件
input_json = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3.log"  # 原始 JSON 文件
output_json = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector3_gt0.log"  # 过滤后的 JSON 文件

filter_litmus_sequences(blacklist_file, input_json, output_json)