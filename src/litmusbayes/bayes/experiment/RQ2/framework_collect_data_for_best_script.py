import subprocess
import time
import sys

# ================= 配置任务队列 =================
# 在这里配置你的多次运行参数，每一项是一个字典
tasks = [

    {
        "stat_log": "./log_record_best_post_final_kmeans_cross_15_factor_1_banana.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_15_factor_1.json"
    },
    {
        "stat_log": "./log_record_best_post_final_kmeans_cross_15_factor_1_2_banana.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_15_factor_1_2.json"
    },
    {
        "stat_log": "./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_2.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2.json"
    },
    {
        "stat_log": "./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_3.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2.json"
    },
    {
        "stat_log": "./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_4.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2.json"
    },
    {
        "stat_log": "./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_5.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2.json"
    },
    {
        "stat_log": "./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_6.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2.json"
    },
    {
        "stat_log": "./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_7.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2.json"
    },
    {
        "stat_log": "./final_results/log_record_best_post_final_kmeans_cross_10_factor_2_banana_8.log",
        "json_path": "best_params_recommendation_robust_final_kmeans_cross_10_factor_2.json"
    },

]


def run_batch():
    total_tasks = len(tasks)
    print(f"Starting batch processing for {total_tasks} tasks...")
    print("-" * 50)

    for i, task in enumerate(tasks):
        stat_log = task["stat_log"]
        json_path = task["json_path"]

        print(f"[Task {i + 1}/{total_tasks}] Executing...")
        print(f"  Stat Log: {stat_log}")
        print(f"  JSON Path: {json_path}")

        # 构建命令行命令，假设你上一个文件保存为 runner.py
        # 如果你的环境使用 python 替代 python3，请将 "python3" 改为 "python"
        cmd = [
            "python3", "framework_collect_data_for_best_for_script.py",
            "--stat_log", stat_log,
            "--json_path", json_path
        ]

        try:
            # subprocess.run 会同步阻塞，直到 runner.py 执行完毕
            result = subprocess.run(cmd, check=True)
            print(f"[Task {i + 1}/{total_tasks}] Finished successfully.")

        except subprocess.CalledProcessError as e:
            # 如果 runner.py 内部崩溃或被 kill，会捕获到这个异常
            print(f"[Task {i + 1}/{total_tasks}] Failed with return code {e.returncode}.")
            # 如果你希望只要有一个任务失败就彻底停止整个脚本，可以取消下面这行的注释：
            # sys.exit(1)

        except KeyboardInterrupt:
            # 允许你使用 Ctrl+C 安全中断整个批处理
            print("\nBatch processing interrupted by user (Ctrl+C). Exiting...")
            sys.exit(0)

        # 如果当前不是最后一个任务，则休眠 10 秒
        if i + 1 < total_tasks:
            print("Sleeping for 10 seconds before the next run...")
            time.sleep(10)
            print("-" * 50)

    print("-" * 50)
    print("All tasks processed!")


if __name__ == "__main__":
    run_batch()