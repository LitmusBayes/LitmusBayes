import os
import json


def extract_cycle_info(folder1, folder2, output_json):
    """
    遍历 folder1，如果文件不在 folder2 中，则提取文件中 Cycle= 后的字符串，
    并保存为 JSON。
    """

    # 1. 获取 folder2 中的所有文件名，存为集合以便快速比对 (O(1) 复杂度)
    if os.path.exists(folder2):
        files_in_folder2 = set(os.listdir(folder2))
    else:
        print(f"警告: 文件夹2 ({folder2}) 不存在，将处理文件夹1中的所有文件。")
        files_in_folder2 = set()

    extracted_data = []

    # 2. 检查 folder1 是否存在
    if not os.path.exists(folder1):
        print(f"错误: 文件夹1 ({folder1}) 不存在。")
        return

    # 3. 遍历 folder1
    print(f"正在处理 {folder1} ...")
    count_processed = 0
    count_skipped = 0

    for filename in os.listdir(folder1):
        file_path1 = os.path.join(folder1, filename)

        # 确保处理的是文件而不是子文件夹
        if os.path.isfile(file_path1):

            # 4. 如果文件在 folder2 中也出现，则跳过
            if filename in files_in_folder2:
                count_skipped += 1
                continue

            # 5. 读取文件并提取 Cycle
            try:
                with open(file_path1, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()

                    found = False
                    for line in lines:
                        # 查找以 Cycle= 开头的行
                        if line.strip().startswith("Cycle="):
                            # 提取等号后面的部分，并去除首尾空白
                            cycle_str = line.split("=", 1)[1].strip()
                            extracted_data.append(cycle_str)
                            found = True
                            break  # 找到后停止读取该文件

                    if found:
                        count_processed += 1

            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")

    # 6. 保存到 JSON 文件
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        print(f"\n处理完成!")
        print(f"跳过文件数 (在文件夹2中存在): {count_skipped}")
        print(f"提取条目数: {count_processed}")
        print(f"结果已保存至: {output_json}")

    except Exception as e:
        print(f"写入 JSON 文件时出错: {e}")


# --- 配置区域 ---
if __name__ == "__main__":
    # 请在这里修改你的实际文件夹路径
    folder_1_path = "/home/whq/Desktop/code_list/perple_test/all_litmus_naive"  # 待遍历的文件夹
    folder_2_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_naive"  # 用于比对过滤的文件夹
    output_file = "cycles.json"  # 结果输出文件

    extract_cycle_info(folder_1_path, folder_2_path, output_file)