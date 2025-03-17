import ast


def merge_vectors(file1_path, file2_path, output_path):
    # 打开两个输入文件和一个输出文件
    with open(file1_path, 'r', encoding='utf-8') as f1, \
            open(file2_path, 'r', encoding='utf-8') as f2, \
            open(output_path, 'w', encoding='utf-8') as out:

        # 同时读取两个文件的每一行 (zip会以较短的文件为准停止)
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            line2 = line2.strip()

            # 跳过空行
            if not line1 or not line2:
                continue

            try:
                # 1. 解析第一行 (提取最后两位)
                # 使用 split(':', 1) 将 "Label" 和 "[数据]" 分开
                label1, data_str1 = line1.split(':', 1)
                vec1 = ast.literal_eval(data_str1)  # 将字符串转换成列表

                # 2. 解析第二行 (将被追加的目标)
                label2, data_str2 = line2.split(':', 1)
                vec2 = ast.literal_eval(data_str2)

                # 3. 提取文件1向量的最后两位
                suffix = vec1[-2:]

                # 4. 拼接到文件2向量的末尾
                vec2.extend(suffix)

                # 5. 写入新文件
                # 保持原来的格式 Label:[1, 2, 3...]
                # 注意：Python列表转字符串默认会有空格，例如 [1, 2]
                out.write(f"{label2}:{str(vec2)}\n")

            except Exception as e:
                print(f"处理行时出错: {e}")
                print(f"File 1: {line1}")
                print(f"File 2: {line2}")


# --- 使用说明 ---
# 将文件名替换为你实际的文件名
file1 = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive/litmus_vector.log'  # 提供后缀的数据源
file2 = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive/litmus_vector1.log'  # 要被追加的主体
output = '/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector1.log'  # 结果保存位置

# 如果你需要生成测试文件来运行代码，请取消下面几行的注释
# with open(file1, 'w') as f: f.write("LB+addr+data-wsi-rfi-data:[12, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\n")
# with open(file2, 'w') as f: f.write("LB+addr+data-wsi-rfi-data:[6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\n")

merge_vectors(file1, file2, output)
print(f"处理完成，结果已保存至 {output}")