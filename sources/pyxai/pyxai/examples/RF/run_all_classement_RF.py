import os
import subprocess
import concurrent.futures
import gc
import sys
from concurrent.futures import ProcessPoolExecutor

# Path to the folder containing the datasets
datasets_dir = os.path.abspath("../../../../../datasets")
types_list = [100]

# Create all combinations (dataset, type)
tasks = []
for filename in os.listdir(datasets_dir):
    if filename.endswith(".csv"):
        dataset_name = filename[:-4]  # Remove .csv
        dataset_path = os.path.join(datasets_dir, dataset_name)
        for t in types_list:
            tasks.append((dataset_name, dataset_path, t))

# Function to run a task with a timeout
def run_task(dataset_name, dataset_path, t):
    print(f"Running: {dataset_path} with types={t} (unlimited)")

    # Disable garbage collector to avoid memory management crashes
    gc.disable()

    try:
        subprocess.run(
            [
                "python3",
                "main_classementrules_RF.py",
                f"-dataset={dataset_path}",
                f"-types={t}"
            ],
            # timeout=64800,  # 18 hours
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error for dataset={dataset_name}, types={t}: {e}")
    except Exception as e:
        print(f"Unexpected error for dataset={dataset_name}, types={t}: {e}")
    print("--------------------------------------------------")

    # Re-enable garbage collector after execution
    gc.enable()

# Launching in parallel with ThreadPool
max_threads = 16  # ← Adjust this number according to your CPU

with ProcessPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(run_task, name, path, t) for name, path, t in tasks]
    concurrent.futures.wait(futures)

# Clean exit after execution
sys.exit(0)
