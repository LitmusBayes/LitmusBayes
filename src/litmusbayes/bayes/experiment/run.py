import subprocess
import sys
import os
import datetime


def main():
    # Define tasks using a list of dictionaries to specify the directory and script path
    tasks = [
        # Scripts for plotting and tables in the ../plot directory
        {"dir": "../plot", "path": "plot_RQ1_final_sort_by_default.py"},
        {"dir": "../plot", "path": "plot_RQ2_final_without_sort_by_default.py"},
        {"dir": "../plot", "path": "table_RQ1.py"},
        {"dir": "../plot", "path": "table_RQ2.py"},
        {"dir": "../plot", "path": "table_RQ3_1.py"},
        {"dir": "../plot", "path": "table_RQ3_2.py"},

        # Scripts for vector embeddings and testing in the ./RQ3 directory
        {"dir": "./RQ3/vector", "path": "vec_emb_graph_rgcn_test_gt0_and_save.py"},
        {"dir": "./RQ3/vector", "path": "vec_emb_n_gram_test_gt0_and_save.py"},
        {"dir": "./RQ3/vector", "path": "vec_emb_n_gram_test_gt0_and_save_n_gram.py"},
        {"dir": "./RQ3/vector", "path": "vec_emb_test_gt0_and_save.py"},
        {"dir": "./RQ3/vector", "path": "vec_emb_two_tower_test_split_gt0_save.py"},

        # Scripts for Bayes check framework in the ./RQ3 directory
        {"dir": "./RQ3/vector", "path": "framework_for_bayes_check_dnn.py"},
        {"dir": "./RQ3/vector", "path": "framework_for_bayes_check_for_multi.py"},
        {"dir": "./RQ3/vector", "path": "framework_for_bayes_check_gnn.py"},
        {"dir": "./RQ3/vector", "path": "framework_for_bayes_check_hand.py"},
        {"dir": "./RQ3/vector", "path": "framework_for_bayes_check_n_gram.py"},
        {"dir": "./RQ3/vector", "path": "framework_for_bayes_check_n_gram_dnn.py"},
        {"dir": "./RQ3/vector", "path": "framework_for_bayes_check_two_tower.py"},
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

            # Check if the directory exists
            if not os.path.isdir(script_dir):
                log_and_print(f"❌ Error: Directory '{script_dir}' not found. Skipping {script_name}.")
                continue

            # Construct the full path to check if the file exists
            script_full_path = os.path.join(script_dir, script_name)

            if not os.path.isfile(script_full_path):
                log_and_print(f"⚠️ Warning: File '{script_full_path}' not found. Skipping.")
                continue

            log_and_print(f"▶️ Running: [{script_dir}] -> {script_name}")

            try:
                # Execute the script in its respective directory (cwd=script_dir)
                # capture_output=True intercepts stdout and stderr to write them to the .log file
                result = subprocess.run(
                    [sys.executable, script_name],
                    cwd=script_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Write standard output to the log file
                if result.stdout:
                    log_file.write(result.stdout)

                log_and_print(f"✅ Successfully completed: {script_name}\n")

            except subprocess.CalledProcessError as e:
                # Write standard output and errors to the log file if execution fails
                if e.stdout:
                    log_file.write(e.stdout)
                if e.stderr:
                    log_file.write(e.stderr)

                log_and_print(f"❌ Execution failed: {script_name} (Exit code: {e.returncode})\n")

                # Uncomment the following line if you want to abort the entire batch process upon a single failure
                # break
            except Exception as e:
                log_and_print(f"❌ Unknown error occurred: {e}\n")

        log_and_print("-" * 50 + f"\n🎉 All tasks finished! Outputs are saved in {log_filename}")


if __name__ == "__main__":
    main()