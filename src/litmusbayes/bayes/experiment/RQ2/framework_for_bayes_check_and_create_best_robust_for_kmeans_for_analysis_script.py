import subprocess
import time
import sys

# ================= 配置任务队列 =================
# 配置你需要跑的各种组合，包含 output_file, new_chip_file 和 factor
tasks = [
    {
        "output_file": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2_final_10.json",
        "new_chip_file": "log_record_init_banana_important_kmeans_10_norm.log.validation_cache.jsonl",
        "factor": 2.0,
        "cluster": "cluster_results_final_all_10/cluster_centers_all.json",
    }
]

def run_batch():
    total_tasks = len(tasks)
    print(f"Starting batch processing for {total_tasks} tasks...")
    print("-" * 50)

    for i, task in enumerate(tasks):
        output_file = task["output_file"]
        new_chip_file = task["new_chip_file"]
        factor = task["factor"]
        cluster = task["cluster"]

        print(f"[Task {i + 1}/{total_tasks}] Executing...")
        print(f"  Output JSON:   {output_file}")
        print(f"  New Chip File: {new_chip_file}")
        print(f"  Factor:        {factor}")
        print(f"  Cluster:        {cluster}")
        # 构建命令行命令，请确保 "robust_selector.py" 是你第一步修改后的主程序文件名
        # 如果你环境默认是 python 即可，则保留 "python"，如果是 "python3" 请自行修改
        cmd = [
            "python", "framework_for_bayes_check_and_create_best_robust_for_kmeans_for_analysis_for_script.py",
            "--output_file", output_file,
            "--new_chip_file", new_chip_file,
            "--factor", str(factor),  # 命令行参数需转换为字符串
            "--cluster", str(cluster)
        ]

        try:
            # subprocess.run 会同步阻塞，直到本次任务执行完毕
            result = subprocess.run(cmd, check=True)
            print(f"[Task {i + 1}/{total_tasks}] Finished successfully.")

        except subprocess.CalledProcessError as e:
            print(f"[Task {i + 1}/{total_tasks}] Failed with return code {e.returncode}.")
            # 如果希望只要报错就终止整个批处理，取消下面这行注释
            # sys.exit(1)

        except KeyboardInterrupt:
            # 允许使用 Ctrl+C 安全中断整个批处理
            print("\nBatch processing interrupted by user (Ctrl+C). Exiting...")
            sys.exit(0)

        # 如果当前不是最后一个任务，则休眠 10 秒释放/缓冲资源
        if i + 1 < total_tasks:
            print("Sleeping for 10 seconds before the next run...")
            time.sleep(10)
            print("-" * 50)

    print("-" * 50)
    print("All tasks processed!")

if __name__ == "__main__":
    run_batch()