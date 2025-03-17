import json
import os
import shutil


def copy_litmus_files(json_path, source_dir, target_dir):
    # 1. 如果目标文件夹不存在，创建它
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print(f"已创建新文件夹: {target_dir}")
        except OSError as e:
            print(f"创建文件夹失败: {e}")
            return

    # 统计数据
    success_count = 0
    fail_count = 0

    print(f"开始从 {source_dir} 复制文件到 {target_dir}...")

    # 2. 读取 JSON 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # 逐行读取（针对 JSON Lines 格式）
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                try:
                    data = json.loads(line)
                    litmus_name = data.get('litmus')

                    if not litmus_name:
                        print(f"行 {line_num} 缺少 'litmus' 字段，跳过。")
                        continue

                    # 3. 构建源文件路径
                    # 尝试情况 A: 文件名完全匹配 (例如 "test")
                    src_file_path = os.path.join(source_dir, litmus_name)

                    # 尝试情况 B: 如果找不到，尝试加上 .litmus 后缀 (例如 "test.litmus")
                    if not os.path.isfile(src_file_path):
                        src_file_path_with_ext = src_file_path + ".litmus"
                        if os.path.isfile(src_file_path_with_ext):
                            src_file_path = src_file_path_with_ext
                        else:
                            # 确实找不到文件
                            print(f"[失败] 找不到文件: {litmus_name} (在行 {line_num})")
                            fail_count += 1
                            continue

                    # 4. 执行复制
                    # 获取最终的文件名（包含后缀）
                    final_filename = os.path.basename(src_file_path)
                    dst_file_path = os.path.join(target_dir, final_filename)

                    shutil.copy2(src_file_path, dst_file_path)  # copy2 保留文件元数据（时间戳等）
                    # print(f"[成功] 复制: {final_filename}") # 如果文件太多，可以注释掉这行减少刷屏
                    success_count += 1

                except json.JSONDecodeError:
                    print(f"行 {line_num} JSON 格式错误，跳过。")
                    continue

    except FileNotFoundError:
        print(f"错误: 找不到 JSON 文件: {json_path}")
        return

    # 5. 总结
    print("-" * 30)
    print(f"处理完成。")
    print(f"成功复制: {success_count} 个文件")
    print(f"失败/未找到: {fail_count} 个文件")
    print(f"文件保存在: {target_dir}")


# ================= 配置区域 =================
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache4_norm.jsonl"
# 1. 你的 JSON 文件路径
json_file = cache_file_path  # 举例: "C:/Projects/litmus_scores.json"

# 2. 原始存放 litmus 文件的文件夹
source_folder = litmus_path  # 举例: "C:/All_Litmus_Tests/"

# 3. 你想创建的新文件夹名字
new_target_folder = "selected_litmus_tests"

# ===========================================

if __name__ == "__main__":
    # 确保路径由用户根据实际情况填写
    # 为了演示，你可以直接运行，它会报错找不到文件，请修改上面的路径
    copy_litmus_files(json_file, source_folder, new_target_folder)