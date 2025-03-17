import json
import os


def process_clusters(cluster_path, score_path, output_path):
    # 1. 加载聚类数据
    if not os.path.exists(cluster_path):
        print(f"错误: 找不到文件 {cluster_path}")
        return
    with open(cluster_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)

    # 2. 加载分数数据
    if not os.path.exists(score_path):
        print(f"错误: 找不到文件 {score_path}")
        return
    with open(score_path, 'r', encoding='utf-8') as f:
        scores = json.load(f)

    results = {}

    # 3. 遍历每个聚类
    for cluster_id, info in clusters.items():
        members = info.get("members", [])

        best_member = None
        max_score = -1.0  # 假设分数都是正数，如果可能有负数可设为 -float('inf')

        # 在成员中寻找分数最高的
        for member in members:
            # 获取该成员的分数，如果分数文件中没有，则默认为 0
            current_score = scores.get(member, 0)

            if current_score > max_score:
                max_score = current_score
                best_member = member

        # 4. 记录结果
        # 你可以根据需要调整保存的格式
        results[cluster_id] = {
            "best_litmus_test": best_member,
            "score": max_score,
            "old_representative": info.get("representative")
        }

    # 5. 保存结果到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"处理完成！结果已保存至: {output_path}")


if __name__ == "__main__":
    # 请确保以下文件名与你的实际文件名一致
    CLUSTER_FILE = './cluster_results/cluster_centers.json'  # 聚类文件名
    SCORE_FILE = '../RQ1/baseline_scores.json'  # 分数文件名
    OUTPUT_FILE = 'best_representatives.json'  # 输出文件名

    process_clusters(CLUSTER_FILE, SCORE_FILE, OUTPUT_FILE)