from decisionNode import DecisionNode
import numpy as np
from pyxai import Learning, Explainer, Tools, Builder
import pandas as pd
from pysat.solvers import Glucose3
import random
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def correct_instance(model, X_test1, y_test1):
    s = 0
    predictions = []
    for instance in X_test1:
        predicted_label = model.predict_instance(instance)
        predictions.append(predicted_label)
        s += 1

    correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

    correct = correct_predictions
    return correct


def precision(model, X_test1, y_test1):
    s = 0
    predictions = []
    for instance in X_test1:
        predicted_label = model.predict_instance(instance)
        predictions.append(predicted_label)
        s += 1

    correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

    accuracy = correct_predictions / len(X_test1)
    return accuracy


def trasforme_tuple_to_binaire(tupl, dt_model):
    s = []
    # print(tupl)
    for n in tupl:
        # print(n)
        # print(bt_model.map_id_binaries_to_features[abs(n)])
        # s.append(tuple(bt_model.map_id_binaries_to_features[abs(n)]) + (True if n < 0 else False,))
        # s.append(bt_model.map_features_to_id_binaries[n])
        is_inside = False
        for e in dt_model.map_features_to_id_binaries:
            # print(dt_model.map_features_to_id_binaries)
            if e[0] == n:
                s.append((dt_model.map_features_to_id_binaries[e])[0])
                is_inside = True
            elif e[0] == -n:
                s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                is_inside = True
        if is_inside is False:
            # print("n not found",n)
            s.append((abs(n), Builder.GT, 0.5, True if n < 0 else False))
    return s


def trasforme_list_tuple_to_binaire(tupl, dt_model):
    s = []
    # print(tupl)
    for k in tupl:
        for n in k:
            # print(n)
            # print(bt_model.map_id_binaries_to_features[abs(n)])
            # s.append(tuple(bt_model.map_id_binaries_to_features[abs(n)]) + (True if n < 0 else False,))
            # s.append(bt_model.map_features_to_id_binaries[n])
            is_inside = False
            for e in dt_model.map_features_to_id_binaries:
                # print(dt_model.map_features_to_id_binaries)
                if e[0] == n:
                    s.append((dt_model.map_features_to_id_binaries[e])[0])
                    is_inside = True
                elif e[0] == -n:
                    s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                    is_inside = True
            if is_inside is False:
                # print("n not found",n)
                s.append((abs(n), Builder.GT, 0.5, True if n < 0 else False))
    return s


def list_to_tuple_pairs(lst):
    if len(lst) % 2 != 0:
        raise ValueError("La liste doit contenir un nombre pair d'éléments.")

    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]


Tools.set_verbose(1)
glucose = Glucose3()
# I load the dataset
path = Tools.Options.dataset
path2 = Tools.Options.types
data = pd.read_csv(path)
name = path
# Dividing the DataFrame into training, testing, and validation sets
train_df, validation_df = train_test_split(data, test_size=0.3, random_state=42)
train_df.columns = data.columns
validation_df.columns = data.columns
# Save the DataFrames to CSV files
train_df.to_csv('train_data.csv', index=False)
validation_df.to_csv('validation_data.csv', index=False)
# best_parameters = DecisionNode.tuning('train_data.csv')
bt_learner = Learning.Xgboost('train_data.csv', learner_type=Learning.CLASSIFICATION)  # 70%
# I create a xgboost model: the expert
bt_model = bt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, seed=0)
instance, prediction = bt_learner.get_instances(bt_model, n=1)
# I need an explainer BT
bt_explainer = Explainer.initialize(bt_model, instance, features_type=path2)
# I need the theory.... Currently, we collect clauses related to a binarised instance

# I need to collect the theory related to boolean variables....
for clause in bt_model.get_theory(bt_explainer.binary_representation):
    glucose.add_clause(clause)

binarized_training = []
raw_validation = []
label_validation = []
binarized_validation = []
nb_features = len(bt_explainer.binary_representation)  # nb binarized features
print('uidjhik')
# Iterating through the training set to binarize it
for i, instance in enumerate(bt_learner.data):
    bt_explainer.set_instance(instance)
    binarized_training.append([0 if l < 0 else 1 for l in bt_explainer.binary_representation] + [bt_learner.labels[i]])

# Iterating through the validation set to binarize it
for i, instance in validation_df.iterrows():
    bt_explainer.set_instance(instance[:-1])
    raw_validation.append(instance[:-1])
    label_validation.append(instance[-1])
    binarized_validation.append([0 if l < 0 else 1 for l in bt_explainer.binary_representation] + [instance[-1]])

training_data = pd.DataFrame(binarized_training, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
training_data.to_csv('training_data.csv', index=False)
dt_learner = Learning.Scikitlearn(training_data, learner_type=Learning.CLASSIFICATION)
# We create K fold cross validation models
# Sauvegarder le DataFrame dans un fichier CSV


# optimisied configuration
# best_parameters2 = DecisionNode.tuning2('train_data.csv')
# dt_models = dt_learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT,seed=0,**best_parameters2)


# default configuration
dt_models = dt_learner.evaluate(method=Learning.K_FOLDS, n_models=2, output=Learning.DT,
                                seed=0)  # , **best_parameters2)

inst2, pred2 = dt_learner.get_instances(dt_models[0], n=1)
print("2")
feature_names = dt_learner.get_details()[0]['feature_names']
all_scikit = dt_learner.get_raw_models()
# Declare the lists and dictionaries that we will use
trees = []
number_of_nodes_decision_tree = []
number_of_nodes_decision_tree0 = []
reasons_with_predictions_dict = {}
reasons_with_predictions_dict1 = {}  # Dictionary to store the reasons
tuple_of_instance_predictions_boosted_tree = []
precision_decision_tree_before_correction_on_the_validation_set = []
X_train1 = []
y_train1 = []
X_test = []
y_test = []
X_train_folds = []
y_train_folds = []
precision_of_the_bosted_tree_on_the_validation_set = []
tree_depth = []
precision_for_each_tree = []
number_of_different_predictions = []
print("3")
# Iterating through the 10 decision tree models created with PyXAI
yo = []
for i, dt_model in enumerate(dt_models):
    yo.append(dt_model)
    print("dj")
    # I take scikitLearn model
    clf = all_scikit[i]
    number_of_nodes_for_a_single_tree0 = clf.tree_.node_count
    # Access the decision tree
    tree = clf.tree_
    # Get the decision tree in tuple form
    tree_tuple = DecisionNode.parse_decision_tree(tree, feature_names)
    transformed_tree = DecisionNode.transform_tree(tree_tuple)
    # Simplify the obtained decision tree
    simplified_tree = DecisionNode.simplify_tree_theorie(transformed_tree, glucose, [])
    trees.append(simplified_tree)
    depth_for_a_single_tree = clf.get_depth()
    # depth_for_a_single_tree=DecisionNode.tree_depth(transformed_tree)
    tree_depth.append(depth_for_a_single_tree)
    number_of_nodes_for_a_single_tree = DecisionNode.count_nodes(transformed_tree)
    number_of_nodes_decision_tree.append(number_of_nodes_for_a_single_tree0)
    number_of_nodes_decision_tree0.append(number_of_nodes_for_a_single_tree)
    # I collect  all instances from training set
    instances_dt_training = dt_learner.get_instances(dt_model, n=None, indexes=Learning.TRAINING, details=True)
    # I collect  all instances from test set
    instances_dt_test = dt_learner.get_instances(dt_model, n=None, indexes=Learning.TEST, details=True)
    X_train1 = []
    y_train1 = []
    X_1 = []
    y_1 = []
    ert = Explainer.initialize(dt_model)
    # Store instances and their labels in the lists X1 and Y1 of the test set
    for instance_dict in instances_dt_test:
        instance_dt = instance_dict["instance"]
        label_dt = instance_dict["label"]
        ert.set_instance(instance_dt)
        X_1.append(instance_dt)
        y_1.append(label_dt)
    # Store instances and their labels in the lists X_train1 and y_train1 of the training set that we will use for retraining
    for instance_dicti in instances_dt_training:
        instance_dt = instance_dicti["instance"]
        label_dt = instance_dicti["label"]
        X_train1.append(instance_dt)
        y_train1.append(label_dt)
    X_train_folds.append(X_train1)
    y_train_folds.append(y_train1)
    # Calculate the accuracy of each decision tree
    # precision_for_a_single_tree=DecisionNode.precision(transformed_tree, X_train1, y_train1)
    precision_for_a_single_tree = DecisionNode.precision(transformed_tree, X_1, y_1)
    precision_for_each_tree.append(precision_for_a_single_tree * 100)
    reasons_with_predictions = []
    reasons_with_predictions1 = []
    single_tuple_instance_prediction = []
    nb = 0
    z = 0
    instancemalclasee = []
    sufficient_reason = []
    X_test1 = []
    y_test1 = []
    # Store instances and their labels in the lists X_test1 and y_test1 of the validation set
    for id_instance, instance_dict in enumerate(binarized_validation):
        instance_dt = instance_dict[:-1]
        label_dt = instance_dict[-1]
        X_test1.append(instance_dt)
        prediction_dt = DecisionNode.classify(simplified_tree, instance_dt)
        bt_explainer.set_instance(raw_validation[id_instance])
        y_test1.append(
            bt_explainer.target_prediction)  # y_test1 is the label, and it is considered as the prediction of the decision tree with 100% confidence
        # Display the instances
        # print("instance_bt:", [0 if l < 0 else 1 for l in bt_explainer.binary_representation])
        assert [0 if l < 0 else 1 for l in bt_explainer.binary_representation] == list(instance_dt), "ca va pas!"
        if (bt_explainer.target_prediction == label_dt):
            z += 1  # However, we still calculate the number of times the decision tree had a correct prediction with the real label of the validation set
        if prediction_dt != bt_explainer.target_prediction:
            ert.set_instance(instance_dt)
            # Store misclassified instances
            instancemalclasee.append(instance_dt)
            # Extract a specific explanation from the decision tree
            tree_specific_reason = bt_explainer.tree_specific_reason(n_iterations=50)
            # Propagate the obtained tree-specific explanations (to avoid creating impossible instances in the retraining)
            propagations = glucose.propagate(tree_specific_reason)
            assert (propagations[0])
            tree_specific_reason1 = propagations[1]
            tree_specific_reason1 = tuple(tree_specific_reason1)
            # Obtain a specific explanation and its prediction in tuple form
            reason_with_prediction1 = (tree_specific_reason1, bt_explainer.target_prediction)
            reasons_with_predictions1.append(reason_with_prediction1)
            reason_with_prediction = (tree_specific_reason, bt_explainer.target_prediction)
            reasons_with_predictions.append(reason_with_prediction)
            # Retrieve instances where the decision tree and the boosted tree give different predictions, and keep the boosted tree's prediction
            tuple_dataa = (tuple(instance_dt), bt_explainer.target_prediction)
            single_tuple_instance_prediction.append(tuple_dataa)
            nb += 1  # Calculate the number of times the boosted tree and decision tree give different predictions
    precision_for_a_single_bosted_tree = (z / len(X_test1)) * 100
    precision_of_the_bosted_tree_on_the_validation_set.append(precision_for_a_single_bosted_tree)
    number_of_different_predictions.append(nb)
    # Store the specific explanation and its prediction in a dictionary for rectification
    reasons_with_predictions_dict[f'reasons_with_predictions{i}'] = reasons_with_predictions
    # Store the propagated specific explanation and its prediction in a dictionary for retraining
    reasons_with_predictions_dict1[f'reasons_with_predictions1{i}'] = reasons_with_predictions1
    X_test.append(X_test1)  # Store all instances from all decision trees
    y_test.append(y_test1)  # Store all labels from all decision trees (which is actually the label of the boosted tree)
    tuple_of_instance_predictions_boosted_tree.append(single_tuple_instance_prediction)
    locf = DecisionNode.precision(simplified_tree, X_test1,
                                  y_test1)  # Calculate the accuracy of the decision tree before correction

    precision_decision_tree_before_correction_on_the_validation_set.append(locf)
print(precision_decision_tree_before_correction_on_the_validation_set)
unique_reasons_with_predictions1 = {}  # Dictionary to store the reasons
unique_reasons_with_predictions = {}
print("4")
# Keep only unique rules, meaning no repetition of rules
for i in range(len(trees)):
    unique_reasons_with_predictions1_i = list(set(reasons_with_predictions_dict1[f'reasons_with_predictions1{i}']))
    unique_reasons_with_predictions1[f'unique_reasons1_{i}'] = unique_reasons_with_predictions1_i

    unique_reasons_with_predictions_i = list(set(reasons_with_predictions_dict[f'reasons_with_predictions{i}']))
    unique_reasons_with_predictions[f'unique_reasons_{i}'] = unique_reasons_with_predictions_i

# Iterate through each list of tuple_of_instance_predictions_boosted_tree
for i, l in enumerate(tuple_of_instance_predictions_boosted_tree):
    # Use a temporary set to track unique tuples
    unique_tuples = set()
    unique_list = []

    # Iterate through the tuples in the current list
    for tup in l:
        # If the tuple is not already in the unique set, add it to the unique list and the set
        if tup not in unique_tuples:
            unique_tuples.add(tup)
            unique_list.append(tup)

    # Replace the original list with the unique list without duplicates
    tuple_of_instance_predictions_boosted_tree[i] = unique_list
# Transform rules in the form of conditions into values: if X1 >= 0.5, assign value 1; otherwise, assign value -1
results = []  # Create an empty list to store the results
results1 = []
data_frames = []
for i in range(len(tuple_of_instance_predictions_boosted_tree)):
    # Create an empty list to store the results of the current iteration
    z = []
    zi = []
    for j in range(len(unique_reasons_with_predictions[f'unique_reasons_{i}'])):
        result = unique_reasons_with_predictions[f'unique_reasons_{i}'][j]
        z.append(result)  # Add the result to the list z
        re = unique_reasons_with_predictions1[f'unique_reasons1_{i}'][j]
        zi.append(re)
    results.append(z)  # Add the list z to the list of results
    results1.append(zi)
    # Put the rules into a data frame for each tree
    # Create a list to store the DataFrames

    # Create a DataFrame from the list z

    df = pd.DataFrame(zi, columns=['sufficient_reason_numeric',
                                   'prediction'])  # Remplacez 'Column1' et 'Column2' par les noms de colonnes appropriés

    # Add the DataFrame to the list of DataFrames
    data_frames.append(df)

# Now you have a list of lists 'results' where each element is a list of results for an iteration of the outer loop.
# We put the rules into a data frame for each tree, filling columns not in the rule with random values


# Create a binary DataFrame with columns for instances and predictions
binary_data_frames = []

for i, df in enumerate(data_frames):
    df_binary = pd.DataFrame(columns=feature_names)
    # Iterate through each row of the DataFrame "df" containing sufficient reasons and associated predictions
    for index, row in df.iterrows():
        # Retrieve the prediction associated with the sufficient reason
        y = row["prediction"]

        # Initialize the values of instances X to random values between 0 and 1
        X = np.random.rand(nb_features)
        # Modify the values of instances X based on the sufficient reason
        for feature in row["sufficient_reason_numeric"]:
            if feature < 0:
                X[abs(feature) - 1] = 0
            else:
                X[feature - 1] = 1

        # Add the values of instances X and the prediction y to a new row of the binary DataFrame
        df_binary.loc[len(df_binary)] = np.concatenate((X, [y]))
    binary_data_frames.append(df_binary)

    # Display the binary DataFrame

# Create a list to store the DataFrames of X_test
X_test_dataframes = []
nouvelles_colonnes = {ancien_nom: f"X_{ancien_nom + 1}" for ancien_nom in range(nb_features)}
nouvelles_colonness = {ancien_nom: f"y" for ancien_nom in range(nb_features)}
# Loop through each list in X_test
for x_test_list in X_test:
    # Create a DataFrame from the current list
    x_test_df = pd.DataFrame(x_test_list)
    x_test_df.rename(columns=nouvelles_colonnes, inplace=True)

    # Add the DataFrame to the list
    X_test_dataframes.append(x_test_df)

# X_test_dataframes will now contain a list of DataFrames, one for each list in X_test
# Create a list to store the DataFrames of y_test

y_test_dataframes = []

# Loop through each list in y_test
for y_test_list in y_test:
    # Create a DataFrame from the current list
    y_test_df = pd.DataFrame(y_test_list)
    y_test_df.rename(columns=nouvelles_colonness, inplace=True)
    # Add the DataFrame to the list
    y_test_dataframes.append(y_test_df)

# y_test_dataframes will now contain a list of DataFrames, one for each list in y_test

# Create a list to store the DataFrames of X_train_folds

X_train_folds_dataframes = []

# Loop through each list in X_train_folds
for X_train_list in X_train_folds:
    # Create a DataFrame from the current list
    X_train_df = pd.DataFrame(X_train_list)
    X_train_df.rename(columns=nouvelles_colonnes, inplace=True)
    # Add the DataFrame to the list
    X_train_folds_dataframes.append(X_train_df)

# X_train_folds_dataframes will now contain a list of DataFrames, one for each list in X_train_folds

# Create a list to store the DataFrames of y_train_folds

y_train_folds_dataframes = []

# Loop through each list in y_train_folds
for y_train_list in y_train_folds:
    # Create a DataFrame from the current list
    y_train_df = pd.DataFrame(y_train_list)
    y_train_df.rename(columns=nouvelles_colonness, inplace=True)
    # Add the DataFrame to the list
    y_train_folds_dataframes.append(
        y_train_df)  # y_train_folds_dataframes will now contain a list of DataFrames, one for each list in y_train_folds

################################################################################
# Here, we will perform correction through Rectification
################################################################################


# To use rectification, we need to retrieve a list of rules that give predictions of 0 and 1 for each tree
resultats_dict = {}

# Using rectification with the extracted rules for each tree
for i in range(len(results)):
    resultat_i = results[i]
    resultat_0, resultat_1 = DecisionNode.split_list(resultat_i)
    # Store the results in the dictionary
    resultats_dict[f"resultat_{i}"] = {"resultat_0": resultat_0, "resultat_1": resultat_1}

# Declare the lists that we will use to store the results
precision_decision_tree_after_rectification_for_all_trees = []
time_rectification_for_each_rule = []
number_of_nodes_for_all_trees = []
depth_rectification_for_each_rule = []
number_of_different_predictions_in_rectification_in_all_trees = []

# Loop to perform rectification on the 10 decision trees
for b in range(len(results)):
    print(b)
    dt_model = yo[b]
    length_of_the_instance = len(X_test[b])
    number_of_different_predictions_in_rectification = [number_of_different_predictions[b]]
    depth_rectification_ = [dt_model.depth()]
    precision_decision_tree_after_rectification = [
        precision_decision_tree_before_correction_on_the_validation_set[b] * 100]
    number_of_nodes = [dt_model.n_nodes()]
    time_unwind = []
    tree_rectified = trees[b]

    # Use b as an index to access the results of the outer loop
    result_key = f"resultat_{b}"
    resultat_1_b = resultats_dict[result_key]["resultat_1"]
    resultat_0_b = resultats_dict[result_key]["resultat_0"]

    # reason = tuple([(2, Builder.GT, 0.5, True)] + [r for r in reason][1:])
    # model = explainer.rectify(conditions=reason, label=1, tests=True)

    # i=trasforme_tuple_to_binaire(resultat_1_b[0][0],bt_model)
    # print(i)
    ert.set_instance(inst2)
    print("ICI", dt_model.to_CNF(inst2))
    ert = Explainer.initialize(dt_model)
    ths = bt_model.get_theory(bt_explainer.binary_representation)
    print("ths", ths)
    theorie = trasforme_list_tuple_to_binaire(ths, dt_model)
    # theorie_clause=list_to_tuple_pairs(theorie)
    theorie_clause = ert.condi(conditions=theorie)
    theorie_clause = list_to_tuple_pairs(theorie_clause)
    print("######################")
    print("theorie", theorie)
    print("transforme", theorie_clause)
    print("Nodes ", dt_model.n_nodes())
    for j in resultat_1_b:
        i = trasforme_tuple_to_binaire(j[0], dt_model)
        # print("j[0]",j[0])
        # print("i",i)
        # yup=ert.condi(conditions=theorie)
        # print("theorie2",yup)
        # exit(0)
        start_time = time.time()
        dt_model = ert.rectify(conditions=i, label=1, tests=False, theory_cnf=theorie_clause)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        precision_tree_rectified = precision(dt_model, X_test[b], y_test[b])
        numbre_of_instance_correct = correct_instance(dt_model, X_test[b], y_test[b])
        numbre_ofdifferentprediction = length_of_the_instance - numbre_of_instance_correct
        total_nodes = dt_model.n_nodes()
        tree_depth_rectification__ = dt_model.depth()
        number_of_nodes.append(total_nodes)
        depth_rectification_.append(tree_depth_rectification__)
        time_unwind.append(elapsed_time)
        number_of_different_predictions_in_rectification.append(numbre_ofdifferentprediction)
        # dt_model = ert.rectify(conditions=i,label=1)
        precision_decision_tree_after_rectification.append(precision_tree_rectified * 100)

        print("Now: ", dt_model.n_nodes())
        ert.set_instance(inst2)
        print(dt_model.to_CNF(inst2))
    exit(1)
    for i in resultat_0_b:
        #  print("i[0]",i[0])
        i = trasforme_tuple_to_binaire(i[0], dt_model)
        #  print("ii",i)
        #  exit(0)
        start_time = time.time()
        dt_model = ert.rectify(conditions=i, label=0, tests=False, theory_cnf=theorie_clause)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        precision_tree_rectified = precision(dt_model, X_test[b], y_test[b])
        numbre_of_instance_correct = correct_instance(dt_model, X_test[b], y_test[b])
        numbre_ofdifferentprediction = length_of_the_instance - numbre_of_instance_correct
        total_nodes = dt_model.n_nodes()
        tree_depth_rectification__ = dt_model.depth()
        number_of_nodes.append(total_nodes)
        depth_rectification_.append(tree_depth_rectification__)
        precision_decision_tree_after_rectification.append(precision_tree_rectified * 100)
        time_unwind.append(elapsed_time)
        number_of_different_predictions_in_rectification.append(numbre_ofdifferentprediction)

    precision_decision_tree_after_rectification_for_all_trees.append(precision_decision_tree_after_rectification)
    number_of_nodes_for_all_trees.append(number_of_nodes)
    depth_rectification_for_each_rule.append(depth_rectification_)
    time_rectification_for_each_rule.append(time_unwind)
    number_of_different_predictions_in_rectification_in_all_trees.append(
        number_of_different_predictions_in_rectification)

data_ = {
    "dataset name": name,
    "accuracy_of_the_boosted_tree_on_test_set": bt_learner.get_details()[0]["metrics"]["accuracy"],
    "accuracy_of_the_boosted_tree_on_validation_set": precision_of_the_bosted_tree_on_the_validation_set[0],
    "columns_of_the_non_binarized_data_set": bt_learner.get_details()[0]['feature_names'],
    "columns_of_the_binarized_data_set": dt_learner.get_details()[0]['feature_names'],
    "accuracy_for_each_tree_on_test_set": precision_for_each_tree,
    "accuracy_for_each_decision_tree_before_correction_on_validation_set_relative_to_the_boosted_tree": precision_decision_tree_before_correction_on_the_validation_set,
    "number_of_nodes_for_each_tree_before_correction": number_of_nodes_decision_tree,
    "accuracy_after_retraining_for_each_rule": precisions_retraining,
    "accuracy_after_rectification_for_each_rule": precision_decision_tree_after_rectification_for_all_trees,
    "number_of_nodes_after_rectification_for_each_rule": number_of_nodes_for_all_trees,
    "number_of_nodes_after_retraining_for_each_rule": number_of_nodes_decision_tree_retraining,
    "depth_after_retraining_for_each_rule": depth_retraining_for_each_rule,
    "depth_after_rectification_for_each_rule": depth_rectification_for_each_rule,
    "time_retraining_for_each_rule": time_retraining_for_each_rule,
    "time_rectification_for_each_rule": time_rectification_for_each_rule,
    "number_of_different_instances_rectification": number_of_different_predictions_in_rectification_in_all_trees,
    "number_of_different_instances_retraining": number_of_different_predictions_for_all_decision_tree_retraining,
    "nbinstance_initial": nb_instance_intiale,
    "nbinstance_final": nb_instance_restante,
    "nbinstajoutéalltree_for_each_rule": nbinstajoutéalltree,
    "nbinstsuppriméesalltree_for_each_rule": nbinstsuppriméesalltree
}
import os

# Name of the JSON file where you want to save the data
file_name = name

# Writing the data to the JSON file
with open(file_name + ".json", 'w') as file_json:
    json.dump(data_, file_json)
print("precision_decision_tree_after_rectification_for_all_trees",
      precision_decision_tree_after_rectification_for_all_trees)
# print("precisions_retraining",precisions_retraining[0])
# print("number_of_nodes_decision_tree",number_of_nodes_decision_tree)
# print("number_of_nodes_for_all_trees",number_of_nodes_for_all_trees)
# print("number_of_nodes_decision_tree_retraining1",number_of_nodes_decision_tree_retraining)
# print("number_of_nodes_decision_tree_retraining2",lkmp)
print("depth_rectification_for_each_rule", depth_rectification_for_each_rule)
print("depth_retraining_for_each_rule", depth_retraining_for_each_rule)
# print("tree_depth",tree_depth)
# print("nb_instance_intiale",nb_instance_intiale)
# print("nb_instance_restante",nb_instance_restante)
# print("nbinstajoutéalltree",nbinstajoutéalltree)
# print("nbinstsuppriméesalltree",nbinstsuppriméesalltree)
# print("#############")
# print("accuracy_for_each_tree_on_test_set",precision_for_each_tree)
# print("time_rectification_for_each_rule",time_rectification_for_each_rule)
# print("time_retraining_for_each_rule",time_retraining_for_each_rule)


# print(number_of_nodes_decision_tree_retraining)
# print("euhe",precisions_retraining)
# print("################""")
# print(nbinstsuppriméesalltree[9])
# print(nbinstajoutéalltree[9])
# # print(nb_train)
# # print(lkmp)
# print(precision_decision_tree_after_rectification_for_all_trees)
# print("eeeeeee")
# print(precisions_retraining)
# print('rfikf,f')
# print(v)
# print("ujyghf")
# print(number_of_nodes_decision_tree_retraining)
# print("iojhed")
# print(number_of_nodes_for_all_trees)
# print("##################")
# print(nb_repetition)
