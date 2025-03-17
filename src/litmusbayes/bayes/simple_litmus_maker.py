import os
import shlex
import subprocess
import shutil


def run_litmus_pipeline(source_dir, output_base_dir):
    """
    1. 遍历 source_dir 下的所有 .litmus 文件
    2. 对每个文件执行 litmus7 命令，输出到 output_base_dir/{test_name}
    3. 进入输出目录执行 make
    """

    # 1. 检查输入目录是否存在
    if not os.path.exists(source_dir):
        print(f"❌ 错误: 找不到源文件路径 {source_dir}")
        return

    # 2. 确保输出根目录存在，不存在则创建
    if not os.path.exists(output_base_dir):
        print(f"📂 创建输出根目录: {output_base_dir}")
        os.makedirs(output_base_dir, exist_ok=True)

    print(f"🚀 开始处理: {source_dir} -> {output_base_dir}")

    # 获取目录下所有文件并排序
    files = sorted(os.listdir(source_dir))

    success_count = 0
    fail_count = 0
    skip_count = 0

    for filename in files:
        # 只处理 .litmus 文件
        if filename.endswith(".litmus"):
            print("-" * 60)

            # 获取完整文件路径
            litmus_file_path = os.path.join(source_dir, filename)

            # 获取文件名（不带后缀），用于创建输出子目录
            # 例如: test_01.litmus -> test_01
            test_name = os.path.splitext(filename)[0]

            # 指定该测试生成的输出文件夹路径
            target_output_dir = os.path.join(output_base_dir, test_name)

            print(f"📄 发现文件: {filename}")
            print(f"   -> 目标路径: {target_output_dir}")

            try:
                # ==========================================
                # 第一步: 运行 litmus7 生成代码
                # ==========================================
                print("   🛠️  [Step 1] 正在运行 litmus7...")

                if not os.path.exists(target_output_dir):
                    print(f"   📂 创建目录: {target_output_dir}")
                    os.makedirs(target_output_dir, exist_ok=True)

                # 构建 litmus7 命令参数列表
                # 注意: -ccopts 和 -O2 分开写
                litmus_args = [
                    "litmus7",
                    "-carch", "RISCV",
                    "-limit", "true",
                    "-affinity", "incr1",
                    "-force_affinity", "true",
                    "-mem", "direct",
                    "-barrier", "pthread",
                    "-stride", "1",
                    "-size_of_test", "100",
                    "-number_of_run", "10",
                    "-driver", "C",
                    "-gcc", "riscv64-unknown-linux-gnu-gcc",
                    "-ccopts", "-O2",
                    "-smtmode", "seq",
                    "-smt", "2",
                    "-avail", "4",
                    litmus_file_path,  # 输入文件
                    "-o", target_output_dir  # 输出目录
                ]

                cmd_str = shlex.join(litmus_args)

                full_command = f"eval $(opam env);{cmd_str}"

                print(f"   ⚙️  执行指令: {full_command}")

                # 执行 litmus7
                subprocess.run(full_command, shell=True, check=True, executable="/bin/bash")

                # ==========================================
                # 第二步: 进入生成的目录运行 make
                # ==========================================
                print("   🔨 [Step 2] 正在执行 Make...")

                # 检查 Makefile 是否生成成功
                if os.path.exists(os.path.join(target_output_dir, "Makefile")):
                    # cwd=target_output_dir 确保在生成的目录下运行 make
                    subprocess.run(["make"], cwd=target_output_dir, check=True)
                    print(f"✅ [成功] {test_name} 处理完成")
                    success_count += 1
                else:
                    print(f"❌ [失败] litmus7 执行后未找到 Makefile: {test_name}")
                    fail_count += 1

            except subprocess.CalledProcessError as e:
                # 区分是 litmus7 失败还是 make 失败
                if e.cmd[0] == "litmus7":
                    print(f"❌ [失败] litmus7 生成代码出错")
                else:
                    print(f"❌ [失败] Make 编译出错")
                fail_count += 1
            except Exception as e:
                print(f"❌ [错误] 发生未知错误: {e}")
                fail_count += 1
        else:
            # 非 litmus 文件跳过但不报错
            # print(f"⚠️ [跳过] 非 litmus 文件: {filename}")
            skip_count += 1

    print("=" * 60)
    print(f"🏁 任务结束. 成功: {success_count} | 失败: {fail_count} | 跳过(非litmus): {skip_count}")


if __name__ == "__main__":
    # ---------------- 配置区域 ----------------

    # 1. 输入: 存放 .litmus 文件的文件夹
    litmus_source_dir = "/home/whq/Desktop/code_list/perple_test/all_litmus_naive"

    # 2. 输出: 生成的 C 代码和编译结果存放的根目录
    #    脚本会在这个目录下自动为每个litmus文件创建一个同名文件夹
    output_root_dir = "/home/whq/Desktop/code_list/perdict_for_WMM/test/benchmark/litmus"

    # ----------------------------------------

    run_litmus_pipeline(litmus_source_dir, output_root_dir)