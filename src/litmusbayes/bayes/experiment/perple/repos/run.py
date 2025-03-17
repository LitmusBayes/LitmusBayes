import subprocess
from pathlib import Path

def batch_run_executables(root_directory):
    root_path = Path(root_directory).resolve()
    
    # 遍历指定目录下的所有内容
    for sub_dir in root_path.iterdir():
        # 只处理文件夹
        if sub_dir.is_dir():
            exe_path = sub_dir / "run.exe"
            
            # 检查文件夹下是否存在 run.exe
            if exe_path.exists():
                # 定义同名 log 文件路径（保存在根目录下）
                log_file_path = root_path / f"{sub_dir.name}.log"
                print(f"开始运行: {sub_dir.name} -> 输出至 {log_file_path.name}")
                
                with open(log_file_path, "w", encoding="utf-8") as log_file:
                    try:
                        # 运行 exe 并重定向标准输出和标准错误到 log 文件
                        subprocess.run(
                            [str(exe_path), "-s", "100000"],
                            cwd=sub_dir,                  # 确保 exe 在其所在的文件夹上下文运行
                            stdout=log_file,              # 重定向标准输出
                            stderr=subprocess.STDOUT,     # 将错误输出也合并到标准输出中
                            text=True,
                            check=True,                    # 运行失败时抛出异常
                            timeout=10,
                        )
                        print(f"[{sub_dir.name}] 运行完成。")
                    except subprocess.CalledProcessError as e:
                        print(f"[{sub_dir.name}] 运行报错，退出码: {e.returncode}。详情请查看日志。")
                    except Exception as e:
                        print(f"[{sub_dir.name}] 执行时发生未知错误: {e}")
            else:
                print(f"跳过 [{sub_dir.name}]: 未找到 run.exe")

if __name__ == "__main__":
    # 将此处的路径替换为你实际要遍历的父文件夹路径
    # 使用 '.' 代表当前脚本所在的目录
    target_dir = "." 
    batch_run_executables(target_dir)
