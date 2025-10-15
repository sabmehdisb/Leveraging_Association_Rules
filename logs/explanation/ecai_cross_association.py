import re
import matplotlib.pyplot as plt
import numpy as np
import statistics
import glob
import argparse
import os
import json

def process_log_file(dataset_file, log_file, output_directory):
    results = {}
    # Read the JSON file
    with open(dataset_file, 'r') as file_json:
        data_read = json.load(file_json)
    
    dataset_name = data_read["dataset_name"]
    dataset_name = os.path.basename(dataset_name)
    dataset_name = dataset_name.replace("_", "\_")

    theory_initial = data_read["avg_theory_initial_size"]
    max_rules = data_read["n_max_rules"]
    theory_association_rules = data_read["avg_len_rules_total"]
    len_rules_2 = data_read["avg_len_rules_2"]
    len_rules_3 = data_read["avg_len_rules_3"]
    nb_instances_excluded = data_read["avg_nb_instances_excluded"]
    size_majority_reason_literal_1 = data_read["avg_size_majority_reason_literal_1"]
    size_majority_reason_literal_2 = data_read["avg_size_majority_reason_literal_2"]
    size_majority_reason_feature_1 = data_read["avg_size_majority_reason_feature_1"]
    size_majority_reason_feature_2 = data_read["avg_size_majority_reason_feature_2"]
    count_inf = data_read["avg_reasons_reduced"]
    count_sup = data_read["avg_reasons_increased"]
    count_eq = data_read["avg_reasons_equal"]
    madelaine_time = data_read["avg_madelaine_time"]
    time_reason1 = data_read["avg_majoritary_reason1_time"]
    time_reason2 = data_read["avg_majoritary_reason2_time"]
    n_original_features = data_read["n_original_features"]
    avg_binarized_features = data_read["avg_all_n_binarized_features"]

    return (dataset_name, theory_initial, theory_association_rules, len_rules_2, len_rules_3, nb_instances_excluded,
            size_majority_reason_literal_1, size_majority_reason_literal_2, size_majority_reason_feature_1,
            size_majority_reason_feature_2, count_inf, count_sup, count_eq, madelaine_time,
            time_reason1, time_reason2, max_rules, n_original_features, avg_binarized_features)

def format_float(value):
    # Format a float with two decimal places
    return "{:.2f}".format(value)

def generate_latex_table(dataset_name, theory_initial, theory_association_rules, len_rules_2, len_rules_3, max_rules, latex_table):
    dataset_name_latex = dataset_name
    values = [theory_initial, theory_association_rules, len_rules_2, len_rules_3, max_rules]
    latex_table += f"{dataset_name_latex} & {' & '.join(map(str, values))} \\\\\n"
    return latex_table

def generate_latex_table2(dataset_name, size_feat1, size_feat2, count_inf, n_instances, latex_table2):
    dataset_name_latex = dataset_name

    if size_feat1 != 0:
        percent_size_reduction = (size_feat2 / size_feat1) * 100
    else:
        percent_size_reduction = 100

    percent_decrease = 100 - percent_size_reduction

    if n_instances != 0:
        percent_inf = (count_inf / n_instances) * 100
    else:
        print(f"[Warning] Division by zero avoided (n_instance=0) for dataset")
        percent_inf = 0

    values = [percent_decrease, percent_inf]
    formatted_values = [format_float(value) for value in values]
    latex_table2 += f"{dataset_name_latex} & {' & '.join(map(str, formatted_values))} \\\\\n"
    return latex_table2

def generate_latex_table3(dataset_name, madelaine_time, time_reason1, time_reason2, n_original_features, avg_binarized_features, latex_table3):
    dataset_name_latex = dataset_name
    values = [madelaine_time, time_reason1, time_reason2, n_original_features, avg_binarized_features]
    formatted_values = [format_float(value) for value in values]
    latex_table3 += f"{dataset_name_latex} & {' & '.join(map(str, formatted_values))} \\\\\n"
    return latex_table3

def execlatex(datasets, output_directory):
    latex_table = "\\begin{table}[h!]\n\\begin{center}\n\\footnotesize\n\\begin{tabular}{lrrrrrr}\n\\hline\n"
    latex_table += "Dataset&ITH&RTH&SR2&SR3&maxrules\\\\\n\\hline\n"

    latex_table2 = "\\begin{table}[h!]\n\\centering\n\\footnotesize\n\\begin{tabular}{lrrr}\n\\hline\n"
    latex_table2 += "Dataset&PSF&PNinf\\\\\n\\hline\n"

    latex_table3 = "\\begin{table}[h!]\n\\begin{center}\n\\footnotesize\n\\begin{tabular}{lrrrrrr}\n\\hline\n"
    latex_table3 += "Dataset&MT&TR1&TR2&OF&BF\\\\\n\\hline\n"

    for dataset in sorted(datasets):
        log_file = dataset
        print("Processing file:", log_file)

        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            with open(log_file, "r") as file:
                lines = file.readlines()
                if len(lines) >= 4:
                    (dataset_name, theory_initial, theory_association_rules, len_rules_2, len_rules_3,
                     nb_instances_excluded, size_lit1, size_lit2, size_feat1, size_feat2, count_inf, count_sup, count_eq,
                     madelaine_time, time_reason1, time_reason2, max_rules, n_original_features, avg_binarized_features) = process_log_file(dataset, log_file, output_directory)

                    n_instances = count_inf + count_sup + count_eq
                    latex_table = generate_latex_table(dataset_name, theory_initial, theory_association_rules, len_rules_2, len_rules_3, max_rules, latex_table)
                    latex_table2 = generate_latex_table2(dataset_name, size_feat1, size_feat2, count_inf, n_instances, latex_table2)
                    latex_table3 = generate_latex_table3(dataset_name, madelaine_time, time_reason1, time_reason2, n_original_features, avg_binarized_features, latex_table3)
                else:
                    print("File is empty or lacks statistics:", dataset)
        else:
            print("File is empty or does not exist:", dataset)

    # Close LaTeX tables
    latex_table += "\\hline\n\\end{tabular}\n\\end{center}\n"
    latex_table += "\\caption{Number of the initial theory, the one extracted from association rules.}\n"
    latex_table += "\\label{tab:evolution}\n\\end{table}\n"
    latex_table += "Table \\ref{tab:evolution} shows the number of initial theories (ITH), the total number of globally extracted association rules (RTH), as well as the number of rules of size 2 (SR2) and size 3 (SR3), and maxrules represents the maximum number of extracted association rules.\\\\"

    latex_table2 += "\\hline\n\\end{tabular}\n"
    latex_table2 += "\\caption{Statistics about the extraction AssociationRules.}\n"
    latex_table2 += "\\label{tab:statistics}\n\\end{table}\n"
    latex_table2 += "Table \\ref{tab:statistics} displays the explanation size statistics before and after adding association rules. SL1 and SL2 represent the average size of literals before and after adding the theory, respectively. SF1 and SF2 represent the average size of features before and after adding the theory, respectively. Ninf indicates the number of instances where the size decreases after adding the theory, and Nins indicates the number of selected valid instances with the theory.\\\\"

    latex_table3 += "\\hline\n\\end{tabular}\n\\end{center}\n"
    latex_table3 += "\\caption{Time used in the experiments.}\n"
    latex_table3 += "\\label{tab:datasets}\n\\end{table}\n"
    latex_table3 += "Table \\ref{tab:datasets} shows the time used in the experiments. The MT column indicates the execution time of the Madeleine algorithm. The TR1 column corresponds to the average time required to provide explanations for 100 instances before adding association rules, while the TR2 column represents the average time required to provide explanations for 100 instances after adding association rules.\\\\"

    # Print the final LaTeX tables
    print(latex_table)
    print(latex_table2)
    print(latex_table3)

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

# Run the main function in the logs directory
execute_main_in_logs_directory()
