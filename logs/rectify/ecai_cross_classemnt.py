import re
import matplotlib.pyplot as plt
import numpy as np
import statistics
import glob
import argparse
import os
import json
import math

def process_log_file(dataset_json, log_file, output_directory):
    results = {}
    # Read the JSON file
    with open(dataset_json, 'r') as file_json:
        data_read = json.load(file_json)
    # Assign values to corresponding variables
    dataset_name = data_read["dataset_name"]
    dataset_name = os.path.basename(dataset_name)
    dataset_name = dataset_name.replace("_", "\_")

    n_instances = data_read["n_instances"]
    n_original_features = data_read["n_original_features"]
    apriori_class_0 = data_read["avg_apriori_classement_0"]
    apriori_class_1 = data_read["avg_apriori_classement_1"]
    nb_rules = data_read["avg_len_rules_total"]
    elapsed_time_aprioris = data_read["mean_elapsed_time_aprioris"]
    n_binarized_features = data_read["avg_all_n_binarized_features"]
    initial_accuracy_ai = data_read["mean_initial_accuracy"]
    final_accuracy_ai = data_read["mean_final_accuracy"]
    initial_f1_score = data_read["mean_initial_f1_score"]
    final_f1_score=data_read["mean_final_f1_score"]
    initial_gmean_score = data_read["mean_initial_gmean_score"]
    final_gmean_score=data_read["mean_final_gmean_score"]
    initial_auc_score = data_read["mean_initial_auc_score"]
    final_auc_score=data_read["mean_final_auc_score"]
    average_time_rectification = data_read["mean_time_rectification"]
    avg_initial_nodes_ai = data_read["mean_initial_node"]
    avg_final_nodes_ai = data_read["mean_final_node"]
    avg_initial_depth_ai = data_read["mean_initial_depth"]
    avg_final_depth_ai = data_read["mean_final_depth"]
    avg_n_rectifications = data_read["avg_n_rectifications"]
    avg_sufficient_reason1_time = data_read.get("avg_sufficient_reason1_time", data_read.get("avg_majoritary_reason1_time"))
    avg_sufficient_reason2_time = data_read.get("avg_sufficient_reason2_time", data_read.get("avg_majoritary_reason2_time"))

    sums = []

    # Loop through all folds
    for fold in data_read["fold_results"]:
        accuracies = fold["time_rectification"]
        total = sum(accuracies)
        sums.append(total)

    # Compute standard deviation
    if len(sums) > 1:
        std_dev = statistics.stdev(sums)
    else:
        std_dev = 0

    max_sum = max(sums) / 100
    std_dev = std_dev / 100
    print(std_dev, max_sum)
    max_sum = str(max_sum) + " (" + str(std_dev) + ")"
    print(f"Fold sums: {sums}")
    print(f"Maximum rectified accuracy sum: {max_sum}")

    # Extract numerical values from strings
    match = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", initial_accuracy_ai)
    match2 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", final_accuracy_ai)
    match3 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", average_time_rectification)
    match4 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", initial_f1_score)
    match5 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", final_f1_score)
    match6 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", initial_gmean_score)
    match7 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", final_gmean_score)
    match8 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", initial_auc_score)
    match9 = re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", final_auc_score)
    match10=re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", avg_sufficient_reason1_time)
    match11=re.match(r"([\d.eE+-]+)\s*\(([\d.eE+-]+)\)", avg_sufficient_reason2_time)
    if match:
        value1 = float(match.group(1))
        value2 = float(match.group(2))
        initial_accuracy_ai = floor_round(value1,2)
        std_dev_initial = floor_round(value2, 2)
        initial_accuracy_ai = f"{initial_accuracy_ai} ({std_dev_initial})"
    if match2:
        value1 = float(match2.group(1))
        value2 = float(match2.group(2))
        final_accuracy_ai = ceil_round(value1, 2)
        std_dev_final = ceil_round(value2, 2)
        final_accuracy_ai = f"{final_accuracy_ai} ({std_dev_final})"
    if match4:
        value1 = float(match4.group(1))
        value2 = float(match4.group(2))
        initial_f1_score = floor_round(value1,2)
        std_dev_initial = floor_round(value2, 2)
        initial_f1_score = f"{initial_f1_score} ({std_dev_initial})"
    if match5:
        value1 = float(match5.group(1))
        value2 = float(match5.group(2))
        final_f1_score = ceil_round(value1, 2)
        std_dev_final = ceil_round(value2, 2)
        final_f1_score = f"{final_f1_score} ({std_dev_final})"
    if match6:
        value1 = float(match6.group(1))
        value2 = float(match6.group(2))
        initial_gmean_score = floor_round(value1, 2)
        std_dev_final = floor_round(value2, 2)
        initial_gmean_score = f"{initial_gmean_score} ({std_dev_final})"
    if match7:
        value1 = float(match7.group(1))
        value2 = float(match7.group(2))
        final_gmean_score = ceil_round(value1, 2)
        std_dev_final = ceil_round(value2, 2)
        final_gmean_score = f"{final_gmean_score} ({std_dev_final})"
    if match8:
        value1 = float(match8.group(1))
        value2 = float(match8.group(2))
        initial_auc_score = floor_round(value1, 2)
        std_dev_final = floor_round(value2, 2)
        initial_auc_score = f"{initial_auc_score} ({std_dev_final})"
    if match9:
        value1 = float(match9.group(1))
        value2 = float(match9.group(2))
        final_auc_score = ceil_round(value1, 2)
        std_dev_final = ceil_round(value2, 2)
        final_auc_score = f"{final_auc_score} ({std_dev_final})"
    if match3:
        value1 = float(match3.group(1))
        value2 = float(match3.group(2))
        average_time_rectification =  f"{value1:.2e}"
        std_dev_time =  f"{value2:.2e}"
        average_time_rectification = f"{average_time_rectification} ({std_dev_time})"
    if match10:
        value1 = float(match10.group(1))
        value2 = float(match10.group(2))
        avg_sufficient_reason1_time =  f"{value1:.2e}"
        std_dev_time =  f"{value2:.2e}"
        avg_sufficient_reason1_time = f"{avg_sufficient_reason1_time} ({std_dev_time})"
    if match11:
        value1 = float(match11.group(1))
        value2 = float(match11.group(2))
        avg_sufficient_reason2_time =  f"{value1:.2e}"
        std_dev_time =  f"{value2:.2e}"
        avg_sufficient_reason2_time = f"{avg_sufficient_reason2_time} ({std_dev_time})"
    
    print(std_dev_initial, std_dev_final)

    return (dataset_name, n_instances, n_original_features,
            apriori_class_0, apriori_class_1, nb_rules,
            n_binarized_features, elapsed_time_aprioris,
            avg_initial_nodes_ai, avg_final_nodes_ai,
            avg_initial_depth_ai, avg_final_depth_ai,
            average_time_rectification, initial_accuracy_ai,
            final_accuracy_ai, avg_n_rectifications,initial_f1_score,final_f1_score,initial_gmean_score,final_gmean_score,initial_auc_score,final_auc_score,avg_sufficient_reason1_time,avg_sufficient_reason2_time)

def format_float(value):
    return "{:.2f}".format(value)

def ceil_round(value, decimals=2):
    factor = 10 ** decimals
    return math.ceil(value * factor) / factor
def floor_round(value, decimals=2):
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

def generate_latex_table(dataset_name, initial_f1_score,final_f1_score,initial_gmean_score,final_gmean_score,initial_auc_score,final_auc_score,
                         initial_size, final_size, initial_depth, final_depth,
                         rectification_time, n_rectifications, latex_table):
    dataset_latex = dataset_name
    values = [initial_f1_score,final_f1_score,initial_gmean_score,final_gmean_score,initial_auc_score,final_auc_score, initial_size, final_size,
              initial_depth, final_depth, rectification_time, n_rectifications]
    latex_table += f"{dataset_latex} & {' & '.join(map(str, values))} \\\\\n"
    return latex_table

def generate_latex_table2(dataset_name, apriori_time, apriori_0, apriori_1, latex_table2):
    dataset_latex = dataset_name
    formatted_apriori_time = format_float(apriori_time)
    values = [formatted_apriori_time, apriori_0, apriori_1]
    latex_table2 += f"{dataset_latex} & {' & '.join(map(str, values))} \\\\\n"
    return latex_table2

def generate_latex_table3(dataset_name, n_instances, n_original_features,
                          n_binarized_features, n_rules, latex_table3):
    dataset_latex = dataset_name
    values = [n_instances, n_original_features, n_binarized_features, n_rules]
    latex_table3 += f"{dataset_latex} & {' & '.join(map(str, values))} \\\\\n"
    return latex_table3
def generate_latex_table4(dataset_name, avg_sufficient_reason1_time, avg_sufficient_reason2_time,latex_table4):
    dataset_latex = dataset_name
    values = [avg_sufficient_reason1_time, avg_sufficient_reason2_time]
    latex_table4 += f"{dataset_latex} & {' & '.join(map(str, values))} \\\\\n"
    return latex_table4
def execlatex(datasets, output_directory):
    # Table 1
    latex_table = "\\begin{table}[h!]\n\\begin{center}\n\\footnotesize\n\\begin{tabular}{lrrrrrrrrrrrr}\n\\hline\n"
    latex_table += "Dataset&$|\\mathcal{IF}|$&$|\\mathcal{FF}|$&$|\\mathcal{IG}|$&$|\\mathcal{FG}|$&$|\\mathcal{IA}|$&$|\\mathcal{FA}|$&$|\\mathcal{IN}|$&$|\\mathcal{FN}|$&$|\\mathcal{ID}|$&$|\\mathcal{FD}|$&$|\\mathcal{TR}|$&$|\\mathcal{NR}|$\\\\\n\\hline\n"

    # Table 2
    latex_table2 = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{lrrrr}\n\\hline\n"
    latex_table2 += "Dataset&$|\\mathcal{TA}|$&$|{R0}|$&$|{R01}|$\\\\\n\\hline\n"

    # Table 3
    latex_table3 = "\\begin{table}[h!]\n\\begin{center}\n\\footnotesize\n\\begin{tabular}{lrrrrrr}\n\\hline\n"
    latex_table3 += "Dataset&$|D|$ & $|\\mathcal{A}|$ & $|X|$ & $|R|$&Repository\\\\\n\\hline\n"
    # Table 4
    latex_table4 = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{lcc}\n\\hline\n"
    latex_table4 += "Dataset & $\mathcal{T}_{\mathit{bef}}$ & $\mathcal{T}_{\mathit{aft}}$\\\\\n\\hline\n"
    for dataset in sorted(datasets):
        log_file = dataset
        print("Processing file:", log_file)
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            with open(log_file, "r") as file:
                lines = file.readlines()
                if len(lines) >= 4:
                    data = process_log_file(dataset, log_file, output_directory)
                    latex_table = generate_latex_table(data[0], data[16], data[17],data[18], data[19],data[20], data[21], data[8], data[9], data[10], data[11], data[12], data[15], latex_table)
                    latex_table2 = generate_latex_table2(data[0], data[7], data[3], data[4], latex_table2)
                    latex_table3 = generate_latex_table3(data[0], data[1], data[2], data[6], data[5], latex_table3)
                    latex_table4 = generate_latex_table4(data[0],data[22], data[23], latex_table4)
                else:
                    print("File is empty or does not contain enough stats:", dataset)
        else:
            print("File is empty or does not exist:", dataset)

    latex_table += "\\hline\n\\end{tabular}\n\\end{center}\n"
    latex_table += "\\caption{Evolution of the accuracy of $AI$ and other statistics regarding the random forest's evolution.}\n"
    latex_table += "\\label{tab:evolution2}\n\\end{table}\n"
    latex_table += "Table \\ref{tab:evolution2} shows the median of the initial and final accuracy on the validation set ($|\\mathcal{IA}|$, $|\\mathcal{FA}|$), the median of the initial and final number of nodes ($|\\mathcal{IN}|$, $|\\mathcal{FN}|$), the median of the initial and final depth ($|\\mathcal{ID}|$, $|\\mathcal{FD}|$), and $|\\mathcal{TR}|$ indicates the average cumulative time to perform all RF rectifications.\\\\"

    latex_table2 += "\\hline\n\\end{tabular}\n"
    latex_table2 += "\\caption{Statistics about the extraction rules.}\n"
    latex_table2 += "\\label{tab:statistics2}\n\\end{table}\n"
    latex_table2 += "Table \\ref{tab:statistics2} shows the time required by the algorithm to extract the association rules ($|\\mathcal{TA}|$). ($|{R0}|$ and $|{R01}|$) indicate the number of classification rules that output 0 and 1, respectively.\\\\"

    latex_table3 += "\\hline\n\\end{tabular}\n\\end{center}\n"
    latex_table3 += "\\caption{Description of the datasets used in the experiments.}\n"
    latex_table3 += "\\label{tab:datasets2}\n\\end{table}\n"
    latex_table3 += "Table \\ref{tab:datasets2} shows the number of instances in the training and validation sets ($|\\mathcal{T}|$, $|V|$). $|A|$ indicates the initial accuracy on the test set for each dataset, $|R|$ represents the number of rules extracted using the classification rule extraction algorithm, and $|MR|$ indicates the maximum number of selected rules.\\\\"
    
    latex_table4 += "\\hline\n\\end{tabular}\n"
    latex_table4 += "\\caption{Average computation time (in seconds, over 100 instances) for extracting a sufficient reason for an instance given a decision tree.}\n"
    latex_table4 += "\\label{tab:statistics2}\n\\end{table}\n"

    print(latex_table3)
    print(latex_table)
    print(latex_table2)
    print(latex_table4)
def execute_main_in_logs_directory():
    parser = argparse.ArgumentParser(description="Execute main() in logs directory")
    parser.add_argument("--path", required=True, help="Path to log files directory")
    parser.add_argument("--plot", required=True, help="Path to output directory for plots")

    args = parser.parse_args()

    datasets = []

    current_directory = os.getcwd()

    try:
        os.chdir(args.path)
        log_files = glob.glob("*.json")

        for log_file in log_files:
            datasets.append(log_file)

        if not os.path.exists(args.plot):
            os.makedirs(args.plot)

        execlatex(datasets, args.plot)
    finally:
        os.chdir(current_directory)

# Call the function
execute_main_in_logs_directory()
