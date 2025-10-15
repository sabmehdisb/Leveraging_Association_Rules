import os
import subprocess
import concurrent.futures

# Path to the folder containing the datasets
datasets_dir = os.path.abspath("../../../../../datasets")
types_list = [100, 1000, 10000, 100000]

# Create all combinations (dataset, type)
tasks = []
for filename in os.listdir(datasets_dir):
    if filename.endswith(".csv"):
        dataset_name = filename[:-4]  # Remove .csv
        dataset_path = os.path.join(datasets_dir, dataset_name)
        for t in types_list:
            tasks.append((dataset_name, dataset_path, t))

# Function to run a task with timeout
def run_task(dataset_name, dataset_path, t):
    print(f"Running: {dataset_path} with types={t} (unlimited)")
    try:
        subprocess.run(
            [
                "python3",
                "main_associationrules_RF.py",
                f"-dataset={dataset_path}",
                f"-types={t}"
            ],
            #timeout=64800,  # 18 hours
            check=True
        )
    except subprocess.TimeoutExpired:
        print(f"Timeout exceeded for dataset={dataset_name}, types={t}")
    except subprocess.CalledProcessError as e:
        print(f"Error for dataset={dataset_name}, types={t}: {e}")
    print("--------------------------------------------------")

# Launching in parallel with ThreadPool
max_threads = 16  # ← Adjust this number according to your CPU
with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(run_task, name, path, t) for name, path, t in tasks]
    concurrent.futures.wait(futures)
