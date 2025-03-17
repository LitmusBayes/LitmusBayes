import json
import pandas as pd
import numpy as np
from collections import defaultdict

# 配置文件路径
PRIOR_FILE = 'log_record_best_prior_final_debug_1_banana.log.validation_cache.jsonl'
POSTERIOR_FILE = 'log_record_best_post_final_debug_8_banana.log.validation_cache.jsonl'

import json
import os


def load_max_scores(file_path):
    """读取 JSONL 文件，返回每个 litmus 测试的最高分数及其对应的参数。"""
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found -> {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                name = record.get("litmus")
                score = float(record.get("score", 0.0))
                param = record.get("param", [])

                if name not in data or score > data[name]["score"]:
                    data[name] = {"score": score, "param": param}
            except json.JSONDecodeError:
                continue
    return data


def load_clusters(file_path):
    """读取聚类中心的 JSON 文件。"""
    if not os.path.exists(file_path):
        print(f"Warning: Cluster file not found -> {file_path}")
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_models_with_clusters(prior_path, posterior_path, cluster_path, output_log):
    prior_data = load_max_scores(prior_path)
    posterior_data = load_max_scores(posterior_path)
    clusters = load_clusters(cluster_path)

    # 建立 litmus 名字到 cluster_id 的反向映射字典
    litmus_to_cluster = {}
    for cid, info in clusters.items():
        for member in info.get("members", []):
            litmus_to_cluster[member] = cid

    all_litmus = set(prior_data.keys()).union(set(posterior_data.keys()))

    # 初始化按聚类分组的结果字典
    grouped_results = {}
    for cid, info in clusters.items():
        grouped_results[cid] = {
            "representative": info.get("representative", "Unknown"),
            "prior_only": [],
            "posterior_only": []
        }
    grouped_results["unclustered"] = {
        "representative": "None",
        "prior_only": [],
        "posterior_only": []
    }

    # 分类记录差异
    for name in all_litmus:
        p_score = prior_data.get(name, {}).get("score", -1.0)
        post_score = posterior_data.get(name, {}).get("score", -1.0)

        cid = litmus_to_cluster.get(name, "unclustered")

        if p_score > 0 and post_score <= 0:
            grouped_results[cid]["prior_only"].append({
                "litmus": name,
                "prior_score": p_score,
                "prior_param": prior_data[name]["param"],
                "posterior_score": post_score,
                "posterior_param": posterior_data[name]["param"]
            })
        elif post_score > 0 and p_score <= 0:
            grouped_results[cid]["posterior_only"].append({
                "litmus": name,
                "prior_score": p_score,
                "posterior_score": post_score,
                "prior_param": prior_data[name]["param"],
                "posterior_param": posterior_data[name]["param"]
            })

    # 将结果写入 .log 文件
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write("========== Clustered Litmus Test Comparison Log ==========\n\n")

        # 遍历每个聚类写入日志
        for cid, data in grouped_results.items():
            prior_list = data["prior_only"]
            post_list = data["posterior_only"]

            # 如果该聚类下没有任何差异，直接跳过，保持日志整洁
            if not prior_list and not post_list:
                continue

            if cid == "unclustered":
                f.write(f"=== [Unclustered] Litmus Tests ===\n")
            else:
                f.write(f"=== Cluster {cid} (Representative: {data['representative']}) ===\n")

            if prior_list:
                f.write(f"  --- Triggered ONLY by Prior Model (Count: {len(prior_list)}) ---\n")
                # 按先验分数从高到低排序
                for item in sorted(prior_list, key=lambda i: i["prior_score"], reverse=True):
                    f.write(f"    Litmus: {item['litmus']}\n")
                    f.write(f"      Prior Score: {item['prior_score']} | Param: {item['prior_param']}\n")
                    f.write(f"      Posterior Score: {item['posterior_score']} | Param: {item['posterior_param']}\n")

            if post_list:
                f.write(f"  --- Triggered ONLY by Posterior Model (Count: {len(post_list)}) ---\n")
                # 按后验分数从高到低排序
                for item in sorted(post_list, key=lambda i: i["posterior_score"], reverse=True):
                    f.write(f"    Litmus: {item['litmus']}\n")
                    f.write(f"      Posterior Score: {item['posterior_score']} | Param: {item['posterior_param']}\n")
                    f.write(f"      Prior Score: {item['prior_score']} | Param: {item['prior_param']}\n")

            f.write("\n")

    print(f"聚类对比完成！结果已保存至: {output_log}")


if __name__ == "__main__":
    # 请替换为你的实际文件路径

    CLUSTER_FILE = "cluster_results_final/cluster_centers.json"
    OUTPUT_LOG = "clustered_model_comparison.log"

    compare_models_with_clusters(PRIOR_FILE, POSTERIOR_FILE, CLUSTER_FILE, OUTPUT_LOG)