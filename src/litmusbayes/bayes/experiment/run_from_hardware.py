import subprocess
import sys
import os
import datetime
import shutil  # For deleting directories

def main():
    # ==========================================
    # 1. List of files/directories to delete before running (modify as needed)
    # ==========================================
    files_to_delete = [
        # e.g., "./results/old_data.csv",
        # e.g., "../plot/temp_cache_dir",
        "./RQ1/final_result/bayes_final/log_record_final_1.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_final_2.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_final_3.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_final_4.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_final_5.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_final_6.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_final_7.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_final_8.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_1.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_2.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_3.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_4.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_5.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_6.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_7.log.validation_cache.jsonl",
        "./RQ1/final_result/bayes_final/log_record_init_8.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_1.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_2.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_3.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_4.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_5.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_6.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_7.log.validation_cache.jsonl",
        "./RQ2/final_results/init/log_record_init_banana_final_8.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_1.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_2.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_3.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_4.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_5.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_6.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_7.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_post_final_kmeans_cross_10_factor_2_banana_8.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_1.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_2.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_3.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_4.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_5.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_6.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_7.log.validation_cache.jsonl",
        "./RQ2/final_results/post_final/log_record_best_prior_8.log.validation_cache.jsonl"
    ]

    print("🗑️ Starting to clean up specified old files/directories...")
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the entire directory and its contents
                    print(f"✅ Deleted directory: {file_path}")
                else:
                    os.remove(file_path)      # Delete a single file
                    print(f"✅ Deleted file: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
        else:
            print(f"⚠️ Path does not exist, skipping: {file_path}")
    print("-" * 50)


    # ==========================================
    # 2. Define the list of tasks to execute sequentially
    # ==========================================
    tasks = [

        {"dir": "./RQ1", "path": "framework_collect_data_for_init.py"},
        {"dir": "./RQ1", "path": "framework_for_bayes_check_and_create_best_robust.py"},
        {"dir": "./RQ1", "path": "framework_collect_data_for_best.py"},
        {"dir": "./RQ1/final_result/init", "path": "median.py"},
        {"dir": "./RQ1/final_result/bayes_final", "path": "median.py"},

        {"dir": "./RQ2", "path": "framework_collect_data_for_init.py"},
        {"dir": "./RQ2", "path": "framework_for_bayes_check_and_create_best_robust_for_kmeans_for_analysis_script.py"},
        {"dir": "./RQ2", "path": "framework_collect_data_for_best_script.py"},
        {"dir": "./RQ2/final_results/init", "path": "median.py"},
        {"dir": "./RQ2/final_results/prior_final", "path": "median.py"},
        {"dir": "./RQ2/final_results/post_final", "path": "median.py"},
    ]

    # Setup logging to a .log file to capture all standard output and errors
    log_filename = f"execution_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_filename, "w", encoding="utf-8") as log_file:
        def log_and_print(message):
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

        log_and_print("🚀 Starting batch execution...\n" + "-" * 50)

        for task in tasks:
            script_dir = task["dir"]
            script_name = task["path"]

            if not os.path.isdir(script_dir):
                log_and_print(f"❌ Error: Directory '{script_dir}' not found. Skipping {script_name}.")
                continue

            script_full_path = os.path.join(script_dir, script_name)

            if not os.path.isfile(script_full_path):
                log_and_print(f"⚠️ Warning: File '{script_full_path}' not found. Skipping.")
                continue

            log_and_print(f"▶️ Running: [{script_dir}] -> {script_name}")

            try:
                # subprocess.run is synchronous and blocking.
                # The next loop will not start until the current script finishes execution.
                result = subprocess.run(
                    [sys.executable, script_name],
                    cwd=script_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )

                if result.stdout:
                    log_file.write(result.stdout)

                log_and_print(f"✅ Successfully completed: {script_name}\n")

            except subprocess.CalledProcessError as e:
                if e.stdout:
                    log_file.write(e.stdout)
                if e.stderr:
                    log_file.write(e.stderr)

                log_and_print(f"❌ Execution failed: {script_name} (Exit code: {e.returncode})\n")

                # Uncomment the 'break' below if you want to stop the entire batch process immediately upon a script error
                # break
            except Exception as e:
                log_and_print(f"❌ Unknown error occurred: {e}\n")

        log_and_print("-" * 50 + f"\n🎉 All tasks finished! Outputs are saved in {log_filename}")


if __name__ == "__main__":
    main()