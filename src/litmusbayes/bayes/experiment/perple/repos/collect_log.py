import shutil
import argparse
from pathlib import Path


def gather_log_files(source_dir, dest_dir):
    src_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # 确保目标文件夹存在，如果不存在则自动创建
    dest_path.mkdir(parents=True, exist_ok=True)

    # 使用 rglob 递归查找所有子文件夹中的 .log 文件
    log_files = list(src_path.glob("*.log"))

    if not log_files:
        print(f"在 '{source_dir}' 中没有找到任何 .log 文件。")
        return

    print(f"共找到 {len(log_files)} 个 .log 文件，开始复制...")

    success_count = 0
    for log_file in log_files:
        # 排除目标文件夹内的 .log 文件（防止自己复制自己）
        if dest_path in log_file.parents:
            continue

        try:
            file_name = f'{log_file.stem.split("_")[0]}.log'
            target_file_path = dest_path / file_name

            # 使用 copy2 可以尽可能保留原始文件的元数据
            shutil.copy2(log_file, target_file_path)
            print(f"已复制: {log_file.name} -> {file_name}")
            success_count += 1

        except Exception as e:
            print(f"复制 '{log_file.name}' 时发生错误: {e}")

    print(f"\n操作完成！成功复制了 {success_count} 个文件到 '{dest_path.resolve()}'。")


if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description="收集并复制日志文件")
    parser.add_argument("--DEST_DIR", type=str, required=True, help="日志收集的目标文件夹路径")
    args = parser.parse_args()

    SOURCE_DIR = "."

    # 传入解析后的参数
    gather_log_files(SOURCE_DIR, args.DEST_DIR)