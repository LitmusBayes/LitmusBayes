import os


def list_files_no_extension(source_folder, output_file):
    """
    读取指定文件夹下的所有文件名（不带后缀），并写入输出文件。
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误：文件夹 '{source_folder}' 不存在。")
        return

    try:
        # 获取文件夹内所有内容
        files = os.listdir(source_folder)

        # 排序，让输出更整齐
        files.sort()

        with open(output_file, 'w', encoding='utf-8') as f_out:
            count = 0
            for filename in files:
                # 拼接完整路径用于判断是否为文件
                full_path = os.path.join(source_folder, filename)

                # 只处理文件，忽略子文件夹
                if os.path.isfile(full_path):
                    # os.path.splitext 将文件名和后缀分开，[0] 取文件名部分
                    name_without_ext = os.path.splitext(filename)[0]

                    # 写入文件并换行
                    f_out.write(name_without_ext + '\n')
                    count += 1

        print(f"成功！已提取 {count} 个文件名到 '{output_file}'。")

    except Exception as e:
        print(f"发生错误: {e}")


# ==========================================
# 配置区域：请修改下面的路径
# ==========================================

# 1. 你要统计的文件夹路径 (Windows路径请注意使用双反斜杠 \\ 或前加 r)
target_folder = r'/home/whq/Desktop/code_list/perple_test/perple_json'

# 2. 结果保存的文件路径
result_file = 'can_perple.log'

# 运行函数
if __name__ == '__main__':
    list_files_no_extension(target_folder, result_file)