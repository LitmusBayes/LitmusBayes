import os


def delete_empty_files(directory, dry_run=True):
    """
    遍历指定目录，删除所有大小为 0 的文件。
    :param directory: 要扫描的文件夹路径
    :param dry_run: 如果为 True，只打印不删除；为 False 则执行删除。
    """

    if not os.path.exists(directory):
        print(f"错误: 文件夹 '{directory}' 不存在。")
        return

    print(f"正在扫描文件夹: {directory}")
    if dry_run:
        print("【模式: 试运行 (Dry Run)】不会实际删除文件。")
    else:
        print("【模式: 实际执行】即将删除文件！")
    print("-" * 40)

    deleted_count = 0
    error_count = 0

    # os.walk 会递归遍历所有子文件夹
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                # 检查文件大小是否为 0
                if os.path.getsize(file_path) == 0:
                    if dry_run:
                        print(f"[将删除] {file_path}")
                    else:
                        os.remove(file_path)
                        print(f"[已删除] {file_path}")

                    deleted_count += 1

            except Exception as e:
                print(f"[错误] 处理文件 {file_path} 时出错: {e}")
                error_count += 1

    print("-" * 40)
    print("扫描结束。")
    if dry_run:
        print(f"发现空文件: {deleted_count} 个 (未删除)")
        print("请将脚本中的 DRY_RUN 改为 False 以执行删除。")
    else:
        print(f"成功删除空文件: {deleted_count} 个")

    if error_count > 0:
        print(f"错误次数: {error_count}")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 目标文件夹路径 (例如刚才生成的 litmus_output)
    target_folder = "./litmus_output"

    # 【安全开关】
    # True  = 只打印，不删除 (建议先跑一遍这个)
    # False = 真的删除
    DRY_RUN = False

    delete_empty_files(target_folder, DRY_RUN)