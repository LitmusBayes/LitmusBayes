import os
import re


def extract_cycle_strings(source_folder, output_file):
    results = []

    # 获取文件夹下所有文件
    # 注意：使用 sorted() 是为了避免不同机器上 os.listdir 返回顺序不一致的问题
    try:
        filenames = sorted(os.listdir(source_folder))
    except FileNotFoundError:
        print(f"错误: 找不到文件夹 {source_folder}")
        return

    print(f"正在处理 {len(filenames)} 个文件...")

    for filename in filenames:
        file_path = os.path.join(source_folder, filename)

        # 跳过子文件夹，只处理文件
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # 使用正则查找以 "Cycle=" 开头的行
                # ^Cycle= 匹配行首的Cycle=，(.*)$ 捕获该行剩余所有字符
                match = re.search(r'^Cycle=(.*)$', content, re.MULTILINE)

                if match:
                    # group(1) 是 Cycle= 后面的内容，strip() 去除首尾空格
                    extracted_str = match.group(1).strip()
                    results.append(f"{filename} - {extracted_str}")
                else:
                    # 如果文件中没有 Cycle 字段，可以选择跳过或记录
                    results.append(f"{filename} - Not Found")
                    pass

        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(results))

    print(f"处理完成，结果已保存至: {output_file}")


# --- 配置区域 ---
# 请将此处路径替换为你存放 litmus test 文件的实际文件夹路径
source_directory = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
output_filename = "/home/whq/Desktop/code_list/perple_test/bayes_stat/cycle.txt"

if __name__ == "__main__":
    extract_cycle_strings(source_directory, output_filename)