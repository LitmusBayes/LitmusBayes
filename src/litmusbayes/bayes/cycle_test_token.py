import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

# ================= 配置路径 =================
# 输入：包含 "TestName - PosWW PosWR..." 的文件
INPUT_FILE = "/home/whq/Desktop/code_list/perple_test/bayes_stat/cycle.txt"
# 输出：生成的向量文件，供训练代码读取
OUTPUT_FILE = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"


def generate_and_save_vectors(input_path, output_path):
    print(f"Processing {input_path} ...")

    # 1. 读取并解析数据
    data = []
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    with open(input_path, "r") as f:
        # 一次性读取并处理每一行
        for line in f:
            line = line.strip()
            if not line or ' - ' not in line:
                continue

            # 拆分文件名和Token序列 (只分割第一个 ' - ')
            name, tokens_str = line.split(' - ', 1)

            data.append({
                'litmus_name': name.strip(),
                'token_string': tokens_str.strip()
            })

    if not data:
        print("No valid data found!")
        return

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} litmus tests.")

    # 2. 统计 Token (生成词袋向量)
    # lowercase=False: 保持 Rfi, PosWW 的大小写，不转为小写
    # token_pattern: 匹配所有单词，包括单字母
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)

    # 转换
    X = vectorizer.fit_transform(df['token_string'])

    # 获取特征名 (词表)
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nDetected {len(feature_names)} features (Token Types):")
    print(feature_names)

    # 3. 转换为 Dense 矩阵 (List)
    dense_vectors = X.toarray()

    # 4. 保存为适配训练代码的格式
    # 格式目标: "文件名: [1, 0, 2, ...]"
    print(f"\nSaving vectors to {output_path} ...")

    count = 0
    with open(output_path, "w") as f_out:
        for i in range(len(df)):
            name = df.iloc[i]['litmus_name']
            # 将 numpy array 转为 python list
            vec_list = dense_vectors[i].tolist()

            # 写入文件
            f_out.write(f"{name}:{vec_list}\n")
            count += 1

    print(f"Done! Saved {count} vectors.")

    # 5. (可选) 打印预览
    preview_df = pd.DataFrame(dense_vectors[:5], columns=feature_names)
    preview_df.insert(0, 'TestName', df['litmus_name'][:5])
    print("\n=== Preview (Top 5) ===")
    print(preview_df)


if __name__ == "__main__":
    generate_and_save_vectors(INPUT_FILE, OUTPUT_FILE)