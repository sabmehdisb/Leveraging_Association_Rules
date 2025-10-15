# Imported functions
import apriori_classement_rules
from pyxai import Learning, Explainer, Tools, Builder
from sklearn.model_selection import train_test_split
import pandas as pd
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt
import time
import json
import os
import statistics
import numpy as np

def extract_mean_std(l):
    m = statistics.mean(l)
    s = np.std(l)
    return str(m) + " (" + str(s) + ")"

##############################################################################################################

Tools.set_verbose(0)
glucose = Glucose3()
name = Tools.Options.dataset
n_max_rules = int(Tools.Options.types)
print("n_max_rules:", n_max_rules)

data = pd.read_csv(name + '.csv')
n_original_features = len(data.columns) - 1

# Parameters for cross-validation
k_folds = 10
print(f"Preparing for {k_folds}-fold cross-validation")

# Structures to store results for all folds
all_initial_accuracies = []
all_final_accuracies = []
all_initial_f1_score = []
all_final_f1_score = []
all_n_rectifications = []
all_conflict_percentages = []
all_elapsed_times = []
all_initial_depth = []
all_final_depth = []
all_initial_node = []
all_final_node = []
alltimerectification = []
all_initial_gmean_score = []
all_final_gmean_score = []
all_initial_auc_score = []
all_final_auc_score = []


train_df, validation_data = train_test_split(data, test_size=0.3, random_state=0)

# I prepare the model with cross-validation (K_FOLDS)
rf_learner = Learning.Scikitlearn(train_df, learner_type=Learning.CLASSIFICATION)
rf_models = rf_learner.evaluate(method=Learning.K_FOLDS, output=Learning.RF, seed=0)

all_n_binarized_features = []
fold_results = []

# For each fold in cross-validation
for fold_idx, rf_model in enumerate(rf_models):
    print(f"\n###### FOLD {fold_idx + 1}/{k_folds} ######")
    
    # Retrieve instances for this fold
    instance, prediction = rf_learner.get_instances(rf_model, n=1)
    
    # Initialize an explainer
    rf_explainer = Explainer.initialize(rf_model, instance, features_type=name + '.types')

    instances_dt_training = rf_learner.get_instances(rf_model, n=None, indexes=Learning.TRAINING, details=True)

    # I collect all instances from test set
    ert = Explainer.initialize(rf_model)
    train_data = []
    train_labels = []

    # Store instances and their labels in the lists X_train1 and y_train1 for retraining
    for instance_dicti in instances_dt_training:
        instance_dt1 = instance_dicti["instance"]
        label_dt = instance_dicti["label"]
        train_data.append(instance_dt1)
        train_labels.append(label_dt)

    # Collect clauses related to the binarized instance
    clauses = []
    # Collect the theory related to boolean variables
    for clause in rf_model.get_theory(rf_explainer.binary_representation):
        glucose.add_clause(clause)
        negated_clause = [[-clause[0]], [clause[1]]]
        clauses.append(negated_clause)

    binarized_training = []
    raw_validation = []
    label_validation = []
    binarized_validation = []
    nb_features = len(rf_explainer.binary_representation)  # number of binarized features

    # Iterate over the training set to binarize it
    for i, instance in enumerate(train_data):
        rf_explainer.set_instance(instance)
        binarized_training.append([0 if l < 0 else 1 for l in rf_explainer.binary_representation] + [train_labels[i]])
    
    training_data = pd.DataFrame(binarized_training, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
    n_binarized_features = len(training_data.columns) - 1
    all_n_binarized_features.append(n_binarized_features)

    # Iterate over the validation set to binarize it
    for i, instance in validation_data.iterrows():
        rf_explainer.set_instance(instance.iloc[:-1])
        raw_validation.append(instance.iloc[:-1])
        label_validation.append(int(instance.iloc[-1]))
        binarized_validation.append(
            [0 if l < 0 else 1 for l in rf_explainer.binary_representation] + [int(instance.iloc[-1])]
        )
    labels = rf_learner.labels_to_values(label_validation)

    ##############################################################################################################

    rf_learner1 = Learning.Scikitlearn(training_data, learner_type=Learning.CLASSIFICATION)
    rf_model1 = rf_learner1.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, seed=0)

    # Get model details
    clf = rf_learner1.get_raw_models()[0]  # Get the scikit-learn model
    total_nodes = sum(tree.tree_.node_count for tree in clf.estimators_)

    # Compute average tree depth
    depth_for_all_trees = [tree.tree_.max_depth for tree in clf.estimators_]
    max_depth = max(depth_for_all_trees)
    average_depth = sum(depth_for_all_trees) / len(depth_for_all_trees)

    number_of_nodes_random_forest = rf_model1.n_nodes()
    random_forest_depth = rf_model1.depth()

    # Get the validation data
    X_test = []
    y_test = []
    for id_instance, instance_dict in enumerate(binarized_validation):
        instance_dt = instance_dict[:-1]
        label_dt = labels[id_instance]
        X_test.append(instance_dt)
        y_test.append(label_dt)

    # Compute initial precision on validation set
    precision_before_correction = apriori_classement_rules.precision(rf_model1, X_test, y_test)
    F1_score_before_correction = apriori_classement_rules.f1_score(rf_model1, X_test, y_test)
    G_Mean_before_correction=apriori_classement_rules.gmean_score(rf_model1, X_test, y_test)
    AUC_before_correction=apriori_classement_rules.auc_score_from_binary(rf_model1, X_test, y_test)
    print(f"Initial precision on validation set (Fold {fold_idx + 1}): {precision_before_correction}")
    print(f"Initial F1-score on test set (Fold {fold_idx + 1}): {F1_score_before_correction}")
    print(f"Initial G_Mean on test set (Fold {fold_idx+1}): {G_Mean_before_correction}")
    print(f"Initial AUC on test set (Fold {fold_idx+1}): {AUC_before_correction}")



    #apriori
    ##############################################################################################################
    start_time = time.time()
    info_supports, len_rules_2, len_rules_3, len_rules_total, madelaine_time, rules = apriori_classement_rules.madelaine(training_data, time_limit=3600, n_max_rules=n_max_rules, explainer=rf_explainer)
    end_time = time.time()

    print(f"Fold {fold_idx+1} - len_rules_2:", len_rules_2)
    print(f"Fold {fold_idx+1} - len_rules_3:", len_rules_3)
    print(f"Fold {fold_idx+1} - len_rules_total:", len_rules_total)

    print(f"Fold {fold_idx+1} - max_supports:", info_supports[0])
    print(f"Fold {fold_idx+1} - min_supports:", info_supports[1])
    print(f"Fold {fold_idx+1} - max_supports_instances:", info_supports[2])
    print(f"Fold {fold_idx+1} - min_supports_instances:", info_supports[3])

    elapsed_time_aprioris = (end_time - start_time)
    association_dict = {}
    antecedents = []
    consequents = []
    for antecedent, consequent in rules:
        association_dict[antecedent] = consequent
    association_dict_copy = dict(association_dict)
    new_association_dict = dict(association_dict)

    nb_rules = []
    nb_rules.append(len(new_association_dict))
    class_association_dict = {}
    class_association_dict0 = {}
    y = nb_features + 1
    for antecedent, consequent in new_association_dict.items():
        if (y == consequent):
            class_association_dict[antecedent] = consequent
        if (-y == consequent):
            class_association_dict0[antecedent] = consequent

    print(f"Fold {fold_idx+1} - number of classement rules:", len(class_association_dict) + len(class_association_dict0))
    tuple_of_tuples = [(tuple(key), 1) for key in class_association_dict.keys()]
    tuple_of_tuples0 = [(tuple(key), 0) for key in class_association_dict0.keys()]
    ##############################################################################################################

    # Rectification process
    precisions = [precision_before_correction]
    f1_score=[F1_score_before_correction]
    gmean_score=[G_Mean_before_correction]
    auc_score=[AUC_before_correction]
    number_of_nodes = [number_of_nodes_random_forest]
    depth_rectification = [random_forest_depth]
    time_unwind = []
    n_times_in_conflict_with_validation = 0
    n_rectifications = 0

    # Initialize the explainer for rectification
    ert = Explainer.initialize(rf_model1)
    ths = rf_model.get_theory(rf_explainer.binary_representation)
    theorie = apriori_classement_rules.trasforme_list_tuple_to_binaire(ths, rf_model1)
    theorie_clause = ert.condi(conditions=theorie)
    theorie_clause = apriori_classement_rules.list_to_tuple_pairs(theorie_clause)
    eft=Explainer.initialize(rf_model1)
    elapsed_time_majoritary_reason1 = []
    size_majoritary_reason1=[]
    for i in X_test[:100]:
        start_time = time.time()
        eft.set_instance(i)
        # sufficient_reason = eft.sufficient_reason()
        majoritary_reason = eft.majoritary_reason(n_iterations=1, seed=1)
        end_time = time.time()
        print("time: ", end_time - start_time, "classic reason: ", len(majoritary_reason))
        elapsed_time_majoritary_reason1.append(end_time - start_time)
        size_majoritary_reason1.append(len(majoritary_reason))
    # Apply the rules that predict class 1
    for j in tuple_of_tuples:
        conditions = apriori_classement_rules.trasforme_tuple_to_binaire(j[0], rf_model1)
        start_time = time.time()
        previous_model = rf_model1
        rf_model1 = eft.rectify(conditions=conditions, label=1, tests=False, theory_cnf=theorie_clause)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        precision_tree_rectified = apriori_classement_rules.precision(rf_model1, X_test, y_test)
        f1_score_tree_rectified=apriori_classement_rules.f1_score(rf_model1, X_test, y_test)
        gmean_score_tree_rectified=apriori_classement_rules.gmean_score(rf_model1, X_test, y_test)
        auc_score_tree_rectified=apriori_classement_rules.auc_score_from_binary(rf_model1, X_test, y_test)
        total_node = rf_model1.n_nodes()
        random_forest_depth_rectification = rf_model1.depth()
        
        if precision_tree_rectified < precisions[-1] or f1_score_tree_rectified < f1_score[-1] :
            rf_model1 = previous_model
            n_times_in_conflict_with_validation += 1
        else:
            n_rectifications += 1
            number_of_nodes.append(total_node)
            depth_rectification.append(random_forest_depth_rectification)
            precisions.append(precision_tree_rectified)
            f1_score.append(f1_score_tree_rectified)
            gmean_score.append(gmean_score_tree_rectified)
            auc_score.append(auc_score_tree_rectified)
            time_unwind.append(elapsed_time)
            print(f"Fold {fold_idx+1} - Rule applied (class 1): Precision improved to {precision_tree_rectified}")

    # Apply the rules that predict class 0
    for i in tuple_of_tuples0:
        conditions = apriori_classement_rules.trasforme_tuple_to_binaire(i[0], rf_model1)
        start_time = time.time()
        previous_model = rf_model1
        rf_model1 = eft.rectify(conditions=conditions, label=0, tests=False, theory_cnf=theorie_clause)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        precision_tree_rectified = apriori_classement_rules.precision(rf_model1, X_test, y_test)
        f1_score_tree_rectified=apriori_classement_rules.f1_score(rf_model1, X_test, y_test)
        gmean_score_tree_rectified=apriori_classement_rules.gmean_score(rf_model1, X_test, y_test)
        auc_score_tree_rectified=apriori_classement_rules.auc_score_from_binary(rf_model1, X_test, y_test)
        total_node = rf_model1.n_nodes()
        random_forest_depth_rectification = rf_model1.depth()
        
        if precision_tree_rectified < precisions[-1] or f1_score_tree_rectified < f1_score[-1]:
            rf_model1 = previous_model
            n_times_in_conflict_with_validation += 1
        else:
            n_rectifications += 1
            number_of_nodes.append(total_node)
            depth_rectification.append(random_forest_depth_rectification)
            precisions.append(precision_tree_rectified)
            f1_score.append(f1_score_tree_rectified)
            gmean_score.append(gmean_score_tree_rectified)
            auc_score.append(auc_score_tree_rectified)
            time_unwind.append(elapsed_time)
            print(f"Fold {fold_idx+1} - Rule applied (class 0): Precision improved to {precision_tree_rectified}")

    print("###########################################################################################################")
    eft=Explainer.initialize(rf_model1)
    elapsed_time_majoritary_reason2 = []
    size_majoritary_reason2=[]
    for i in X_test[:100]:
        start_time = time.time()
        eft.set_instance(i)
        # sufficient_reason = eft.sufficient_reason()
        majoritary_reason = eft.majoritary_reason(n_iterations=1, seed=1)
        end_time = time.time()
        print("time: ", end_time - start_time, "classic reason: ", len(majoritary_reason))
        elapsed_time_majoritary_reason2.append(end_time - start_time)
        size_majoritary_reason2.append(len(majoritary_reason))



    print("##########################################################################################################")
    # Calculate the results for this fold
    initial_accuracy = precisions[0]
    final_accuracy = precisions[-1]
    initial_f1_score = f1_score[0]
    final_f1_score = f1_score[-1]
    initial_gmean_score = gmean_score[0]
    final_gmean_score= gmean_score[-1]
    initial_auc_score = auc_score[0]
    final_auc_score= auc_score[-1]
    initial_node=number_of_nodes[0]
    final_node=number_of_nodes[-1]
    initial_depth=depth_rectification[0]
    final_depth=depth_rectification[-1]
    percent_in_conflict_with_validation = (n_times_in_conflict_with_validation / (len_rules_total)) * 100 if len_rules_total > 0 else 0

    print(f"Fold {fold_idx+1} - We pass from {initial_accuracy} to {final_accuracy}")
    print(f"Fold {fold_idx+1} - n_times_in_conflict_with_validation:", n_times_in_conflict_with_validation)
    print(f"Fold {fold_idx+1} - percent_in_conflict_with_validation:", percent_in_conflict_with_validation)
    print(f"Fold {fold_idx+1} - n_rectifications:", n_rectifications)
    print("depth_after_rectification",depth_rectification)
    print("number_of_nodes_after_rectification",number_of_nodes,)
    print("accuracy_after_rectification",precisions)
    print("f1_score_after_rectification",f1_score)
    print("gmean_score_after_rectification",gmean_score)
    print("auc_score_after_rectification",auc_score)
    # Store the results of this fold
    all_initial_accuracies.append(initial_accuracy)
    all_final_accuracies.append(final_accuracy)
    all_initial_f1_score.append(initial_f1_score)
    all_final_f1_score.append(final_f1_score)
    all_initial_gmean_score.append(initial_gmean_score)
    all_final_gmean_score.append(final_gmean_score)
    all_initial_auc_score.append(initial_auc_score)
    all_final_auc_score.append(final_auc_score)
    all_n_rectifications.append(n_rectifications)
    all_conflict_percentages.append(percent_in_conflict_with_validation)
    all_elapsed_times.append(elapsed_time_aprioris)
    all_initial_node.append(initial_node)
    all_final_node.append(final_node)
    all_initial_depth.append(initial_depth)
    all_final_depth.append(final_depth)
    if len(time_unwind) > 0:
        alltimerectification.append(sum(time_unwind))
    else:
        alltimerectification.append(0.0)  
    # Prepare data for JSON output for this fold
    fold_data = {
            "fold_{fold_idx+1}":fold_idx +1 ,
            "len_rules_2": len_rules_2,
            "len_rules_3": len_rules_3,
            "max_supports": info_supports[0],
            "min_supports": info_supports[1],
            "max_supports_instances": info_supports[2],
            "min_supports_instances": info_supports[3],
            "len_rules_total": len_rules_total,
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "n_rectifications": n_rectifications,
            "n_times_in_conflict_with_validation": n_times_in_conflict_with_validation,
            "percent_in_conflict_with_validation": percent_in_conflict_with_validation,
            "number_of_nodes_before_correction": number_of_nodes_random_forest,
            "accuracy_before_correction": precision_before_correction,
            "accuracy_after_rectification": precisions,
            "F1_score_before_correction": F1_score_before_correction,
            "f1_score_after_rectification": f1_score,
            "G_Mean_before_correction": G_Mean_before_correction,
            "gmean_score_after_rectification": gmean_score,
            "AUC_before_correction": AUC_before_correction,
            "auc_score_after_rectification": auc_score,
            "number_of_nodes_after_rectification": number_of_nodes,
            "depth_after_rectification": depth_rectification,
            "time_rectification": time_unwind,
            "elapsed_time_aprioris": elapsed_time_aprioris,
            "apriori_classement_0": len(tuple_of_tuples0),
            "apriori_classement_1": len(tuple_of_tuples),
            "elapsed_time_majoritary_reason2": elapsed_time_majoritary_reason2,
            "elapsed_time_majoritary_reason1": elapsed_time_majoritary_reason1,
            "size_majoritary_reason2":size_majoritary_reason2,
            "size_majoritary_reason1":size_majoritary_reason1,
        
    }
    
    fold_results.append(fold_data)


# Calculate the means and standard deviations over all folds
mean_initial_accuracy = extract_mean_std(all_initial_accuracies)
mean_final_accuracy = extract_mean_std(all_final_accuracies)
mean_initial_f1_score = extract_mean_std(all_initial_f1_score)
mean_final_f1_score = extract_mean_std(all_final_f1_score)
mean_initial_gmean_score = extract_mean_std(all_initial_gmean_score)
mean_final_gmean_score = extract_mean_std(all_final_gmean_score)
mean_initial_auc_score = extract_mean_std(all_initial_auc_score)
mean_final_auc_score = extract_mean_std(all_final_auc_score)
mean_time_rectification=extract_mean_std(alltimerectification)
mean_rectifications = statistics.mean(all_n_rectifications)
mean_conflict_percentage = statistics.mean(all_conflict_percentages)
mean_elapsed_time = statistics.mean(all_elapsed_times)
mean_initial_node = statistics.mean(all_initial_node)
mean_final_node = statistics.mean(all_final_node)
mean_initial_depth = statistics.mean(all_initial_depth)
mean_final_depth = statistics.mean(all_final_depth)
print("\n###### CROSS-VALIDATION RESULTS ######")
print(f"Average initial accuracy: {mean_initial_accuracy}")
print(f"Average final accuracy: {mean_final_accuracy}")
print(f"Average number of rectifications: {mean_rectifications:.2f}")
print(f"Average percentage of conflicts: {mean_conflict_percentage:.2f}%")
print(f"Average execution time Apriori: {mean_elapsed_time:.2f} secondes")
print(f"mean_initial_node: {mean_initial_node:.2f} noeuds")
print(f"mean_final_node: {mean_initial_node:.2f} noeuds")
print(f"mean_initial_depth: {mean_initial_depth:.2f} depth")
print(f"mean_final_depth: {mean_final_depth:.2f} depth")

def compute_avg_std(fold_results, key):
    values = [
        np.mean(f[key]) if f[key] else 0
        for f in fold_results
    ]
    return extract_mean_std(values)

# Compute and display average time for majoritary reason 1
avg_majoritary_reason1_time = compute_avg_std(fold_results, "elapsed_time_majoritary_reason1")
print("Average time for majoritary reason 1 (std):", avg_majoritary_reason1_time)

# Compute and display average time for majoritary reason 2
avg_majoritary_reason2_time = compute_avg_std(fold_results, "elapsed_time_majoritary_reason2")
print("Average time for majoritary reason 2 (std):", avg_majoritary_reason2_time)

# Compute and display average size for majoritary_reason 1
avg_size_majoritary_reason1 = compute_avg_std(fold_results, "size_majoritary_reason1")
print("Average size of majoritary_reason 1 (std):", avg_size_majoritary_reason1)

# Compute and display average size for majoritary_reason 2
avg_size_majoritary_reason2 = compute_avg_std(fold_results, "size_majoritary_reason2")
print("Average size of majoritary_reason 2 (std):", avg_size_majoritary_reason2)
# Prepare final data for JSON output
final_data = {
    
    "dataset_name": name,
    "n_original_features": n_original_features,
    "avg_all_n_binarized_features":np.mean(all_n_binarized_features),
    "n_instances": len(data),
    "n_instances_training": len(train_df),
    "n_instances_validation": len(validation_data),
    "k_folds": k_folds,
    "n_max_rules": n_max_rules,
    "avg_len_rules_2": np.mean([f["len_rules_2"] for f in fold_results]),
    "avg_len_rules_3": np.mean([f["len_rules_3"] for f in fold_results]),
    "avg_len_rules_total": np.mean([f["len_rules_total"] for f in fold_results]),
    "avg_max_supports": np.mean([f["max_supports"] for f in fold_results]),
    "avg_min_supports": np.mean([f["min_supports"] for f in fold_results]),
    "avg_max_supports_instances": np.mean([f["max_supports_instances"] for f in fold_results]),
    "avg_min_supports_instances": np.mean([f["min_supports_instances"] for f in fold_results]),
    "avg_n_rectifications": np.mean([f["n_rectifications"] for f in fold_results]),
    "avg_n_times_in_conflict_with_validation": np.mean([f["n_times_in_conflict_with_validation"] for f in fold_results]),
    "avg_percent_in_conflict_with_validation": np.mean([f["percent_in_conflict_with_validation"] for f in fold_results]),
    "avg_majoritary_reason1_time": avg_majoritary_reason1_time,
    "avg_size_majoritary_reason1": avg_size_majoritary_reason1,
    "avg_majoritary_reason2_time": avg_majoritary_reason2_time,
    "avg_size_majoritary_reason2": avg_size_majoritary_reason2,
    "avg_apriori_classement_0": np.mean([f["apriori_classement_0"] for f in fold_results]),
    "avg_apriori_classement_1": np.mean([f["apriori_classement_1"] for f in fold_results]),
    "mean_initial_accuracy": mean_initial_accuracy,
    "mean_final_accuracy": mean_final_accuracy,
    "mean_initial_f1_score": mean_initial_f1_score,
    "mean_final_f1_score": mean_final_f1_score,
    "mean_initial_gmean_score": mean_initial_gmean_score,
    "mean_final_gmean_score": mean_final_gmean_score,
    "mean_initial_auc_score": mean_initial_auc_score,
    "mean_final_auc_score": mean_final_auc_score,
    "mean_time_rectification":mean_time_rectification,
    "mean_elapsed_time_aprioris": mean_elapsed_time,
    "mean_initial_node":mean_initial_node,
    "mean_final_node":mean_final_node,
    "mean_initial_depth":mean_initial_depth,
    "mean_final_depth":mean_final_depth,
    "fold_results": fold_results
}

print("\n###########################################")
print("Global statistics on", k_folds, "folds:")
for key, value in final_data.items():
    if key != "fold_results":
        print(f"{key}: {value}")

name = name+'_' + str(n_max_rules)
# Writing results to a JSON file
file_name = f"{name}_cross_validation_results_{k_folds}_folds_RF"
with open(file_name + ".json", "w") as file_json:
    json.dump(final_data, file_json, indent=4)

print(f"Results written to {file_name}.json")       

