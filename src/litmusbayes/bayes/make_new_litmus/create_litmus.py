import json
import os
import subprocess
import sys


def generate_litmus_files(input_json, output_dir):
    # 1. 检查输入文件
    if not os.path.exists(input_json):
        print(f"错误: 找不到输入文件 {input_json}")
        return

    # 2. 创建输出文件夹 (如果不存在)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出文件夹: {output_dir}")
        except OSError as e:
            print(f"无法创建输出文件夹: {e}")
            return

    # 3. 读取 JSON 数据
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            cycles = json.load(f)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return

    print(f"开始处理 {len(cycles)} 条数据...")

    success_count = 0
    fail_count = 0

    # 4. 遍历并执行命令
    for i, cycle_str in enumerate(cycles):
        # 去除首尾空白，确保命令整洁
        cycle_clean = cycle_str.strip()

        # 生成文件名：空格替换为下划线
        file_name = cycle_clean.replace(" ", "_") + ".litmus"
        output_path = os.path.join(output_dir, file_name)

        # 构建 Shell 命令
        # 注意：
        # 1. 使用 eval $(opam env) 初始化环境
        # 2. cycle_clean 加上双引号，防止 shell 将其视为多个参数
        # 3. > output_path 将标准输出重定向到文件
        command = f'eval $(opam env); diyone7 -arch=RISC-V {cycle_clean} > {output_path}'

        try:
            # shell=True 允许我们在 shell 中执行包含管道/重定向的复杂命令
            # executable='/bin/bash' 确保使用 bash (opam env 通常对 bash/sh 友好)
            subprocess.run(command, shell=True, check=True, executable='/bin/bash')
            success_count += 1
            print(f"\n[成功] 生成: {cycle_clean}")
            # 可选：打印进度 (每处理10个打印一次)
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(cycles)} ...")

        except subprocess.CalledProcessError as e:
            print(f"\n[失败] 无法生成: {cycle_clean}")
            print(f"命令: {command}")
            print(f"错误信息: {e}")
            fail_count += 1

    # 5. 总结
    print("-" * 30)
    print(f"处理完成！")
    print(f"成功生成: {success_count}")
    print(f"失败: {fail_count}")
    print(f"文件保存在: {output_dir}")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 输入的 JSON 文件路径 (之前合并生成的那个文件)
    json_file_path = "rw_wr_mutated_cycles.json"

    # 输出 litmus 文件的文件夹
    output_folder_path = "/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_output"

    generate_litmus_files(json_file_path, output_folder_path)