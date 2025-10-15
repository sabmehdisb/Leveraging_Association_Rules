import json
from pyxai import Learning, Explainer, Tools
import pandas as pd
from pysat.solvers import Glucose3
from sklearn.model_selection import train_test_split
import apriori_association_rules
import time
import numpy as np
##################################################################################################################################
# Load the dataset

n_max_rules = int(Tools.Options.types)
print("n_max_rules:", n_max_rules)
name = Tools.Options.dataset 
data = pd.read_csv(name+'.csv')
n_original_features = len(data.columns)-1

# Cross-validation parameters
n_folds = 10
glucose = Glucose3()

print("Preparing for", n_folds, "fold cross-validation")

# Create a container to store results for each fold
fold_results = []

# Prepare the model with K-fold cross-validation
dt_learner = Learning.Scikitlearn(data, learner_type=Learning.CLASSIFICATION)
dt_models = dt_learner.evaluate(method=Learning.K_FOLDS, output=Learning.RF, seed=0, n_estimators=1)
allgoodinstance = []
all_n_binarized_features = []

# For each fold in the cross-validation
for fold, dt_model in enumerate(dt_models):
    print(f"\n---------- Processing fold {fold+1}/{n_folds} ----------")
    
    # Retrieve instances for this fold
    instance, prediction = dt_learner.get_instances(dt_model, n=1)
    dt_explainer = Explainer.initialize(dt_model, instance, features_type=name+'.types')
    
    instances_dt_training = dt_learner.get_instances(dt_model, n=None, indexes=Learning.TRAINING, details=True)
    instances_dt_test = dt_learner.get_instances(dt_model, n=None, indexes=Learning.TEST, details=True)
    ert = Explainer.initialize(dt_model)
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []

    # Store instances and their labels in lists for retraining
    for instance_dicti in instances_dt_training:
        instance_dt1 = instance_dicti["instance"]
        label_dt = instance_dicti["label"]
        train_data.append(instance_dt1)
        train_labels.append(label_dt)
        
    for instance_dict in instances_dt_test:
        instance_dt = instance_dict["instance"]
        label_dt = instance_dict["label"]
        ert.set_instance(instance_dt)
        validation_data.append(instance_dt)
        validation_labels.append(label_dt)

    # Binarize the dataset
    nb_features = len(dt_explainer.binary_representation)
    binarized = []
    raw_validation = []
    label_validation = []
    binarized_validation = []
    
    # Binarize training data
    for i, instance in enumerate(train_data):
        dt_explainer.set_instance(instance)
        binarized.append([0 if l < 0 else 1 for l in dt_explainer.binary_representation] + [train_labels[i]])
    
    training_data = pd.DataFrame(binarized, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
    
    # Binarize validation data
    for i, instance in enumerate(validation_data):
        dt_explainer.set_instance(instance)
        raw_validation.append(instance)
        label_validation.append(validation_labels[i])
        binarized_validation.append([0 if l < 0 else 1 for l in dt_explainer.binary_representation] + [validation_labels[i]])
    
    print("Data binarized for fold", fold+1)
    n_binarized_features = len(training_data.columns) - 1
    all_n_binarized_features.append(n_binarized_features)

    # Apriori
    #######################################################################################################################################
    # Use Apriori to extract association rules (excluding the class variable)
    df_filtered = training_data.drop(columns=['y'])
    max_length = 3
    print("Starting madelaine for fold", fold+1)
    len_rules_2, len_rules_3, len_rules_total, madelaine_time, rules = apriori_association_rules.madelaine(df_filtered, time_limit=3600, n_max_rules=n_max_rules, explainer=dt_explainer )
    print("Finished madelaine, time: ", madelaine_time)
    print("len_rules_2:", len_rules_2)
    print("len_rules_3:", len_rules_3)
    print("len_rules_total:", len_rules_total)

    # Display the number of rules generated
    print(f"Number of rules: {len(rules)}")
    print("###########################################")
    theory_association_rules = apriori_association_rules.rules_to_clauses(rules)
    theory_initial = dt_explainer.get_theory()

    print('len theory_initial: ', len(theory_initial))
    print("len theory_association_rules: ", len(theory_association_rules))

    ############################################################################################################
    # Select instances for which to extract majoritary explanations
    print("Selecting instances for fold", fold+1)
    good_instances = []
    glucose = Glucose3()
    nb_instances = 100
    nb_instances_excluded = 0
    
    for i in theory_association_rules:
        glucose.add_clause(i)
    
    for id_instance, instance_dict in enumerate(binarized_validation):
        instance_dt = instance_dict[:-1]
        label_dt = instance_dict[-1]
        dt_explainer.set_instance(raw_validation[id_instance])
        if glucose.propagate(dt_explainer.binary_representation)[0] is False:
            nb_instances_excluded += 1
            continue
        good_instances.append(raw_validation[id_instance])
        if len(good_instances) >= nb_instances:
            break

    len_reason = 0
    treasean = []
    nb_is_not_reason = 0
    majoritary_literal_reason1 = []
    majoritary_feature_reason1 = []
    elapsed_time_majoritary_reason1 = []

    ############################################################################################################
    # Extract majoritary explanations before adding the second theory
    print("Computing standard majoritary explanations for fold", fold+1)
    allgoodinstance.append(len(good_instances))
    for i in good_instances:
        start_time = time.time()
        dt_explainer.set_instance(i)
        reason = dt_explainer.majoritary_reason(n_iterations=100, seed=1)
        majoritary_literal_reason1.append(len(reason))
        majoritary_feature_reason1.append(len(dt_explainer.to_features(reason)))
        if not(dt_explainer.is_majoritary_reason(reason)):
            nb_is_not_reason += 1
        treasean.append(len(reason))
        len_reason += len(reason)
        end_time = time.time()
        print("time: ", end_time - start_time, "classic reason: ", len(reason))
        elapsed_time_majoritary_reason1.append(end_time - start_time)

    moreasen1 = len_reason / len(good_instances) if good_instances else 0

    # Add the second theory to the explainer
    for clause in theory_association_rules:
        dt_explainer.add_clause_to_theory(clause)

    len_reason_theorie2 = 0
    nb_is_not_reason2 = 0
    treasean1 = []
    majoritary_literal_reason2 = []
    majoritary_feature_reason2 = []
    elapsed_time_majoritary_reason2 = []

    print("Computing majoritary explanations with additional theory for fold", fold+1)
    # Extract majoritary explanations after adding the second theory
    for i in good_instances:
        start_time = time.time()
        dt_explainer.set_instance(i)
        reason1 = dt_explainer.majoritary_reason(n_iterations=100, seed=1)
        majoritary_literal_reason2.append(len(reason1))
        majoritary_feature_reason2.append(len(dt_explainer.to_features(reason1)))
        if not(dt_explainer.is_majoritary_reason(reason1)):
            nb_is_not_reason2 += 1
        treasean1.append(len(reason1))
        len_reason_theorie2 += len(reason1)
        print("time: ", time.time() - start_time, "additional theory reason: ", len(reason1), dt_explainer.to_features(reason1))
        end_time = time.time()
        elapsed_time_majoritary_reason2.append(end_time - start_time)

    moreasen2 = len_reason_theorie2 / len(good_instances) if good_instances else 0

    print("Computing statistics for fold", fold+1)
    # Iterate through both lists simultaneously and compare explanation sizes

    count_inf = 0
    count_sup = 0
    count_eq = 0
    for t, t1 in zip(treasean, treasean1):
        if t1 < t:
            count_inf += 1
        elif t1 > t:
            count_sup += 1
        else:  # t1 == t
            count_eq += 1

    # Storing the results of this fold
    fold_data = {
        "fold": fold + 1,
        "theory_initial": len(theory_initial),
        "len_rules_2": len_rules_2,
        "len_rules_3": len_rules_3,
        "len_rules_total": len_rules_total,    
        "nb_instances_excluded": nb_instances_excluded,
        "size_majority_reason_literal_1": majoritary_literal_reason1,
        "size_majority_reason_literal_2": majoritary_literal_reason2,
        "size_majority_reason_feature_1": majoritary_feature_reason1,
        "size_majority_reason_feature_2": majoritary_feature_reason2,
        "number_of_reasons_reduced_after_adding_theory": count_inf,
        "number_of_reasons_increased_after_adding_theory": count_sup,
        "number_of_equal_reasons_after_adding_theory": count_eq,
        "number_of_is_not_reason_before_adding_new_clauses": nb_is_not_reason,
        "number_of_is_not_reason_after_adding_new_clauses": nb_is_not_reason2,
        "elapsed_time_aprioris": madelaine_time,
        "elapsed_time_majoritary_reason1": elapsed_time_majoritary_reason1,
        "elapsed_time_majoritary_reason2": elapsed_time_majoritary_reason2,
    }
    
    fold_results.append(fold_data)

# Calculating global statistics on all folds
global_stats = {
    "dataset_name": name,
    "n_max_rules": n_max_rules,
    "n_folds": n_folds,
    "avg_theory_initial_size": np.mean([f["theory_initial"] for f in fold_results]),
    "avg_len_rules_2": np.mean([f["len_rules_2"] for f in fold_results]),
    "avg_len_rules_3": np.mean([f["len_rules_3"] for f in fold_results]),
    "avg_len_rules_total": np.mean([f["len_rules_total"] for f in fold_results]),
    "avg_nb_instances_excluded": np.mean([f["nb_instances_excluded"] for f in fold_results]),
    "avg_size_majority_reason_literal_1": np.mean([np.mean(f["size_majority_reason_literal_1"]) if f["size_majority_reason_literal_1"] else 0 for f in fold_results]),
    "avg_size_majority_reason_literal_2": np.mean([np.mean(f["size_majority_reason_literal_2"]) if f["size_majority_reason_literal_2"] else 0 for f in fold_results]),
    "avg_size_majority_reason_feature_1": np.mean([np.mean(f["size_majority_reason_feature_1"]) if f["size_majority_reason_feature_1"] else 0 for f in fold_results]),
    "avg_size_majority_reason_feature_2": np.mean([np.mean(f["size_majority_reason_feature_2"]) if f["size_majority_reason_feature_2"] else 0 for f in fold_results]),
    "avg_reasons_reduced": np.mean([f["number_of_reasons_reduced_after_adding_theory"] for f in fold_results]),
    "avg_reasons_increased": np.mean([f["number_of_reasons_increased_after_adding_theory"] for f in fold_results]),
    "avg_reasons_equal": np.mean([f["number_of_equal_reasons_after_adding_theory"] for f in fold_results]),
    "avg_madelaine_time": np.mean([f["elapsed_time_aprioris"] for f in fold_results]),
    "avg_majoritary_reason1_time": np.mean([np.mean(f["elapsed_time_majoritary_reason1"]) if f["elapsed_time_majoritary_reason1"] else 0 for f in fold_results]),
    "avg_majoritary_reason2_time": np.mean([np.mean(f["elapsed_time_majoritary_reason2"]) if f["elapsed_time_majoritary_reason2"] else 0 for f in fold_results]),
    "avg_allgoodinstance":np.mean(allgoodinstance),
    "avg_all_n_binarized_features":np.mean(all_n_binarized_features),
    "n_original_features":n_original_features,
    "fold_results": fold_results
}

print("\n###########################################")
print("Global statistics on", n_folds, "folds:")
for key, value in global_stats.items():
    if key != "fold_results":
        print(f"{key}: {value}")

name = name+'_' + str(n_max_rules)
# # Writing the results to a JSON file
file_name = f"{name}_cross_validation_associationrules_{n_folds}_folds_DT"
with open(file_name + ".json", "w") as file_json:
    json.dump(global_stats, file_json, indent=4)

print(f"Results written in {file_name}.json")