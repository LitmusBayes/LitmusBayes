import os


def main():
    # ================= 配置区域 =================
    # 1. 包含数据的 Log 文件路径 (即你提供的那种格式的文件)
    log_file_path = "/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_vector_new_n_gram.log"

    # 2. Litmus 测试文件所在的文件夹路径
    folder_path = "/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_output"
    # ===========================================

    # 检查路径是否存在
    if not os.path.exists(log_file_path):
        print(f"错误: Log 文件未找到: {log_file_path}")
        return
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹未找到: {folder_path}")
        return

    print(f"正在读取文件夹文件列表: {folder_path} ...")
    # 获取文件夹下所有文件名，存入集合查找更快
    try:
        existing_files = set(os.listdir(folder_path))
    except Exception as e:
        print(f"读取文件夹失败: {e}")
        return

    print(f"正在处理 Log 文件: {log_file_path} ...")

    kept_lines = []
    removed_count = 0
    total_lines = 0

    try:
        # 读取所有行
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)

        # 遍历每一行进行检查
        for line in lines:
            line_content = line.strip()
            if not line_content:
                continue

            # 提取名字：分割第一个冒号
            # 格式例如: "Coe_Fence.r.rwdWR... : W Fence..."
            parts = line_content.split(':', 1)

            if len(parts) >= 1:
                litmus_name = parts[0].strip()

                # 拼接文件名，假设文件夹里的文件后缀是 .litmus
                expected_filename = f"{litmus_name}.litmus"

                # 检查文件是否存在于文件夹中
                if expected_filename in existing_files:
                    # 存在，保留这一行 (注意要把原始的换行符带上，或者重新加)
                    kept_lines.append(line)
                else:
                    # 不存在，计数，不加入 kept_lines 即为删除
                    # print(f"正在移除: {litmus_name} (文件夹中不存在)")
                    removed_count += 1
            else:
                # 格式不对的行，根据需求选择保留或删除，这里选择保留防止误删
                kept_lines.append(line)

        # 将筛选后的内容写回文件
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.writelines(kept_lines)

        print("-" * 30)
        print("处理完成。")
        print(f"原始行数: {total_lines}")
        print(f"移除行数: {removed_count}")
        print(f"剩余行数: {len(kept_lines)}")
        print("-" * 30)

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()