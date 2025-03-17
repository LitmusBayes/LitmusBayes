import os


def delete_files_with_keyword(directory, keyword="X", dry_run=True):
    """
    遍历指定目录，删除文件名中包含特定字符（默认是 "X"）的文件。
    """

    if not os.path.exists(directory):
        print(f"错误: 文件夹 '{directory}' 不存在。")
        return

    print(f"正在扫描文件夹: {directory}")
    print(f"筛选条件: 文件名包含 '{keyword}'")

    if dry_run:
        print("【模式: 试运行 (Dry Run)】不会实际删除文件。")
    else:
        print("【模式: 实际执行】即将删除文件！")
    print("-" * 40)

    deleted_count = 0

    # 遍历文件夹中的所有内容
    # 如果你想递归删除子文件夹里的文件，可以用 os.walk(directory) 替换 os.listdir
    # 这里默认只处理当前层级
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 确保处理的是文件，而不是文件夹
        if os.path.isfile(file_path):

            # 检查文件名是否包含关键字
            if keyword in filename:
                try:
                    if dry_run:
                        print(f"[将删除] {filename}")
                    else:
                        os.remove(file_path)
                        print(f"[已删除] {filename}")

                    deleted_count += 1
                except Exception as e:
                    print(f"[错误] 删除 {filename} 失败: {e}")

    print("-" * 40)
    if dry_run:
        print(f"共发现 {deleted_count} 个包含 '{keyword}' 的文件 (未删除)。")
        print("请将脚本中的 DRY_RUN 改为 False 以执行实际删除。")
    else:
        print(f"成功删除 {deleted_count} 个包含 '{keyword}' 的文件。")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 目标文件夹路径
    target_folder = "./litmus_output"

    # 只有当文件名包含这个字符时才删除
    target_char = "X"

    # 【安全开关】
    # True  = 只打印预览，不删除
    # False = 真的删除
    DRY_RUN = False

    delete_files_with_keyword(target_folder, target_char, DRY_RUN)