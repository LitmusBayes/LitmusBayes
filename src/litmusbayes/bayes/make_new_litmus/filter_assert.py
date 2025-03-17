import os
import subprocess
import re
import json


def find_zero_positive_files(input_folder, output_json):
    """
    遍历文件夹，运行 herd7，记录 Positive: 0 的文件。
    """

    # 1. 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"错误: 文件夹 '{input_folder}' 不存在")
        return

    zero_positive_files = []

    # 获取文件列表
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    total_files = len(files)

    print(f"正在扫描 {input_folder} 下的 {total_files} 个文件...")
    print("-" * 40)

    # 2. 遍历文件
    for idx, filename in enumerate(files):
        file_path = os.path.join(input_folder, filename)

        # 仅处理 .litmus 文件 (可选过滤，防止处理无关文件)
        if not filename.endswith(".litmus"):
            continue

        # 构建命令
        # 注意: 需要确保 herd7 和 riscv.cat 在环境路径中，或者指定完整路径
        # 这里假设 riscv.cat 位于默认位置或当前目录。如果不在，请修改 -model 后的路径
        cmd = f"herd7 -model riscv.cat {file_path}"

        try:
            print(cmd)
            # 执行命令并捕获输出
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True  # 以文本形式获取输出
            )

            output = result.stdout
            print(output)
            # 3. 解析输出中的 Positive: <数字>
            # 典型的 herd7 输出包含: "Positive: 0 Negative: 15"
            match = re.search(r'Positive:\s*(\d+)', output)

            if match:
                positive_count = int(match.group(1))

                # 4. 判断条件：如果 Positive 为 0
                if positive_count == 0:
                    zero_positive_files.append(filename)
                    # 可选：实时打印发现的文件
                    # print(f"[发现] {filename} (Positive: 0)")
            else:
                # 如果没找到 Positive 字段，可能是语法错误或其他问题
                print(f"[警告] 无法解析文件 {filename} 的输出")

        except Exception as e:
            print(f"[错误] 处理文件 {filename} 时出错: {e}")

        # 进度条效果 (每 10 个文件打印一次进度)
        if (idx + 1) % 10 == 0:
            print(f"进度: {idx + 1}/{total_files} ...")

    # 5. 保存结果到 JSON
    try:
        # 排序以便查看
        zero_positive_files.sort()

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(zero_positive_files, f, indent=4, ensure_ascii=False)

        print("-" * 40)
        print("扫描完成！")
        print(f"共扫描文件: {total_files}")
        print(f"Positive 为 0 的文件数: {len(zero_positive_files)}")
        print(f"结果已保存至: {output_json}")

    except Exception as e:
        print(f"写入 JSON 文件出错: {e}")


if __name__ == "__main__":
    # --- 配置区域 ---

    # 待扫描的 litmus 文件夹路径
    target_folder = "./litmus_output"

    # 结果输出的 JSON 文件名
    result_json = "zero_positive_files.json"

    find_zero_positive_files(target_folder, result_json)