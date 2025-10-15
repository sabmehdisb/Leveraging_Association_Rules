import pandas as pd
from itertools import combinations
import time
from mybitset import BitSet
from pyxai.sources.core.tools.encoding import CNF, CNFencoding
from pyxai import Builder

def list_to_tuple_pairs(lst):
    if len(lst) % 2 != 0:
        raise ValueError("The list must contain an even number of elements.")
    
    return [(lst[i], lst[i+1]) for i in range(0, len(lst), 2)]

def trasforme_list_tuple_to_binaire(tupl, rf_model):
    s = []
    for k in tupl:
        for n in k:
            is_inside = False
            for e in rf_model.map_features_to_id_binaries:
                if e[0] == n: 
                    s.append((rf_model.map_features_to_id_binaries[e])[0])
                    is_inside = True
                elif e[0] == -n:
                    s.append(-(rf_model.map_features_to_id_binaries[e])[0])
                    is_inside = True
            if not is_inside:
                s.append((abs(n), Builder.GT, 0.5, True if n < 0 else False))
    return s

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
import math

def gmean_score(model, X_test, y_test):
    # Collect predictions
    predictions = []
    for instance in X_test:
        predicted_label = model.predict_instance(instance)
        predictions.append(predicted_label)
    
    # Calculate confusion matrix components
    true_positives = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 1)
    false_positives = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 0)
    true_negatives = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 0)
    false_negatives = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 1)
    
    # Sensibilité (recall positif)
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Spécificité (recall négatif)
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # G-Mean
    gmean = math.sqrt(sensitivity * specificity)
    
    return gmean
def auc_score_from_binary(model, X_test, y_test):
    # Collect predictions (0 ou 1)
    predictions = []
    for instance in X_test:
        predicted_label = model.predict_instance(instance)
        predictions.append(predicted_label)

    # Calculer TP, FP, TN, FN
    true_positives = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 1)
    false_positives = sum(1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 0)
    true_negatives = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 0)
    false_negatives = sum(1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 1)

    # TPR (Recall) et FPR
    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0

    # Dans le cas binaire, l’AUC ≈ aire sous la courbe reliant (0,0) -> (FPR,TPR) -> (1,1)
    auc = (1 + tpr - fpr) / 2  

    return auc

def f1_score(model, X_test, y_test):
    # Collect predictions
    predictions = []
    for instance in X_test:
        predicted_label = model.predict_instance(instance)
        predictions.append(predicted_label)
    
    # Calculate true positives, false positives, and false negatives
    # This implementation assumes a binary classification (0 and 1)
    true_positives = sum(1 for pred, true_label in zip(predictions, y_test) 
                         if pred == 1 and true_label == 1)
    
    false_positives = sum(1 for pred, true_label in zip(predictions, y_test) 
                          if pred == 1 and true_label == 0)
    
    false_negatives = sum(1 for pred, true_label in zip(predictions, y_test) 
                          if pred == 0 and true_label == 1)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate the F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def trasforme_tuple_to_binaire(tupl, dt_model):
    s = []
    for n in tupl:
        is_inside = False
        for e in dt_model.map_features_to_id_binaries:
            if e[0] == n: 
                s.append((dt_model.map_features_to_id_binaries[e])[0])
                is_inside = True
            elif e[0] == -n:
                s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                is_inside = True
        if not is_inside:
            s.append((abs(n), Builder.GT, 0.5, True if n < 0 else False))
    return s

def convert(antecedent):
    converted_antecedent = []
    for item in antecedent:
        if 'X_' in item:
            converted_antecedent.append(int(item.replace('X_', '')))
        elif 'N_' in item:
            converted_antecedent.append(int(item.replace('N_', '-')))
        else:
            converted_antecedent.append(item)
    return frozenset(converted_antecedent)

def generelise(rule1, rule2, valeur1, valeur2):
    """
    Check if two rules are in conflict
    """
    rule1 = set(rule1)
    rule2 = set(rule2)
    # Check if the rules conditions are the same
    if valeur1 == valeur2:
        if rule1.issubset(rule2):
            return True
    return False

def genereliseclassement(rule1, rule2, valeur1, valeur2):
    """
    Check if two rules are in conflict
    """
    # Check if the rules conditions are the same
    # if valeur1.issubset(valeur2):
    if rule1.issubset(rule2):
        return True
    return False

def remove_element_from_key(dictionary, key, element_to_remove):
    """
    Removes a specific element from a key in a dictionary and creates a new key with the value associated with the original key.

    Args:
        dictionary (dict): The dictionary containing the keys and values.
        key (frozenset): The key to modify.
        element_to_remove: The element to remove from the key.

    Returns:
        dict: The updated dictionary.
    """
    # Check if the key exists in the dictionary
    if key in dictionary:
        # Create a new key without the element to remove
        new_key = tuple(x for x in key if x != element_to_remove)
        # Copy the value associated with the original key
        value = dictionary[key]
        # Remove the old key from the dictionary
        del dictionary[key]
        # Add the new key with the same value
        dictionary[new_key] = value
    return dictionary

# Function to transform the tuples
def rules_to_clauses(rules):
    clauses = []
    for antecedent, consequent in rules:
        # Negation of the first tuple and concatenation with the second
        negated_antecedent = tuple(-x for x in antecedent)
        # Concatenate the transformed first tuple with the second tuple.
        clause = negated_antecedent + (consequent,)
        clauses.append(clause)  
    return clauses


def remove_subsumed(cnf):
    cnf = sorted(cnf, key=lambda clause: len(clause))
    subsumed = [False for _ in range(len(cnf) + 1)]
    flags = [False for _ in range(CNFencoding.compute_max_id_variable(cnf) + 1)]
    for i, clause in enumerate(cnf):
        if subsumed[i]:
            continue
        for lit in clause:
            flags[abs(lit)] = True
        for j in range(i + 1, len(cnf)):
            nLiteralsInside = tuple(flags[abs(lit)] for lit in cnf[j]).count(True)
            if nLiteralsInside == len(clause):
                subsumed[j] = True
        for lit in clause:
            flags[abs(lit)] = False
    return CNF([clause for i, clause in enumerate(cnf) if not subsumed[i]])

#Convert the columns into numbers.
def convert(antecedent):
    converted_antecedent = []
    for item in antecedent:
        if 'X_' in item:
            converted_antecedent.append(int(item.replace('X_', '')))
        elif 'N_' in item:
            converted_antecedent.append(-int(item.replace('N_', '')))
        else:
            converted_antecedent.append(int(item))  # Si l'élément n'a ni 'X_' ni 'N_'
    return converted_antecedent

def generate_candidates(itemsets, length):
    # Generate candidates by combining frequent itemsets.
    return {
        itemsets[i].union(itemsets[j])
        for i in range(len(itemsets))
        for j in range(i+1, len(itemsets))
        if len(itemsets[i].union(itemsets[j])) == length
    }
    
    
    return [frozenset(combination) for combination in combinations(itemsets, length)]

def get_frequent_itemsets_old(transactions, candidates, min_support):
    # Count the occurrences of the candidates in the transactions.
    itemset_counts = {}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                if candidate not in itemset_counts:
                    itemset_counts[candidate] = 0
                itemset_counts[candidate] += 1

    num_transactions = len(transactions)
    frequent_itemsets = {
        itemset for itemset, count in itemset_counts.items()
        if count / num_transactions >= min_support
    }
    itemset_supports = {
        itemset: count / num_transactions
        for itemset, count in itemset_counts.items()
        if count / num_transactions >= min_support
    }
    return frequent_itemsets, itemset_supports

def get_frequent_itemsets(transactions, candidates, min_support):
    # Count the occurrences of the candidates in the transactions.
    #itemset_counts = {itemset: 0 for itemset in candidates}
    itemset_counts = [0]*len(candidates)


    for transaction in transactions:
        for i, candidate in enumerate(candidates):
            if candidate.issubset(transaction):
                itemset_counts[i] += 1
    
    num_transactions = len(transactions)
    
    frequent_itemsets = [
        itemset for i, itemset in enumerate(candidates)
        if itemset_counts[i] / num_transactions >= min_support
    ]


    itemset_supports = {
        itemset: itemset_counts[i] / num_transactions
        for i, itemset in enumerate(candidates)
        if itemset_counts[i] / num_transactions >= min_support
    }

    return frequent_itemsets, itemset_supports


def generate_rules(frequent_itemsets, itemset_supports, min_confidence):
    print("Start generate_rules")
    rules_dict = {}
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
                for subset in map(frozenset, combinations(itemset, len(itemset) - 1)):
                    antecedent = subset
                    consequent = itemset - antecedent
                    if itemset in itemset_supports and antecedent in itemset_supports:
                        confidence = itemset_supports[itemset] / itemset_supports[antecedent]
                        if confidence >= min_confidence:
                            if antecedent in rules_dict:
                                rules_dict[antecedent] = rules_dict[antecedent].union(consequent)
                            else:
                                rules_dict[antecedent] = consequent

    # Generate the rules only if the combination exists in itemset_supports.
    rules = [(antecedent, consequent, itemset_supports[antecedent | consequent] / itemset_supports[antecedent])
             for antecedent, consequent in rules_dict.items() if antecedent | consequent in itemset_supports]
    return rules
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def madelaine(database, time_limit=3600, n_max_rules=200000, explainer=None):
    
    #Compute: key -> value
    # dict_values_0: index_feature -> list of indexes of instances where the index_feature value is 0
    # dict_values_1: index_feature -> list of indexes of instances where the index_feature value is 1 
    total_time = time.time()

    database_tuples = tuple(database.itertuples(index=False, name=None))
    n_instances = len(database_tuples)
    n_features = len(database_tuples[0])
    print("n_instances:", n_instances)
    print("n_features:", n_features)
    dict_values_0 = {i:[] for i in range(1, n_features+1)}
    dict_values_1 = {i:[] for i in range(1, n_features+1)}

    for index_instance, instance in enumerate(database_tuples):
        for index_feature, value in enumerate(instance):
            dict_values_0[index_feature+1].append(index_instance) if value == 0 else dict_values_1[index_feature+1].append(index_instance)

    for i in range(1, n_features+1):
        dict_values_0[i] = set(dict_values_0[i])
        dict_values_1[i] = set(dict_values_1[i])
    
    hash_a_y = [False]*(n_features+1)
    hash_not_a_y = [False]*(n_features+1)
    hash_a_not_y = [False]*(n_features+1)
    hash_not_a_not_y = [False]*(n_features+1)

    hash_a_b = [False]*((n_features+1)*n_features+1)
    hash_not_a_b = [False]*((n_features+1)*n_features+1)
    hash_a_not_b = [False]*((n_features+1)*n_features+1)
    hash_not_a_not_b = [False]*((n_features+1)*n_features+1)


    # Compute all A->B to remove the A and B -> Y that are subsumed 
    candidates = tuple(combinations(range(1, n_features-1), 2)) # n_features-1 to remove y

    for candidate in candidates:
        a, b = candidate[0], candidate[1]
        key = a * b
        support_a_b = dict_values_1[a].intersection(dict_values_1[b])
        support_a_not_b = dict_values_1[a].intersection(dict_values_0[b])
        support_not_a_b = dict_values_0[a].intersection(dict_values_1[b])
        support_not_a_not_b = dict_values_0[a].intersection(dict_values_0[b])

        if len(support_a_not_b) == 0:
            # for a -> b: there is no (a -> not b) in the instances
            hash_a_b[key] = True
        elif len(support_a_b) == 0:
            # for a -> not b: no a -> b
            hash_a_not_b[key] = True
        if len(support_not_a_not_b) == 0:
            # for not a -> b: no not a -> not b
            hash_not_a_b[key] = True
        elif len(support_not_a_b) == 0:
            # for not a -> not b: no not a -> b
            hash_not_a_not_b[key] = True
        


    #for k == 2: generate all a->b rules
    candidates = tuple(combinations(range(1, n_features), 1))

    print("len candidates (k=2):", len(candidates))

    #Test all candidates: test a -> b, not(a) -> b, a -> not(b) and not(a) -> not(b)  
    rules = []
    y = n_features
    n_tests = 0

    for candidate in candidates:
        a = candidate[0]
        
        support_a_y = dict_values_1[a].intersection(dict_values_1[y])
        support_a_not_y = dict_values_1[a].intersection(dict_values_0[y])
        support_not_a_y = dict_values_0[a].intersection(dict_values_1[y])
        support_not_a_not_y = dict_values_0[a].intersection(dict_values_0[y])

        if len(support_a_not_y) == 0:
            # for a -> b: there is no (a -> not b) in the instances
            rules.append((((a,), y), len(support_a_y)))
            hash_a_y[a] = True
        elif len(support_a_y) == 0:
            # for a -> not b: no a -> b
            rules.append((((a,), -y), len(support_a_not_y)))
            hash_a_not_y[a] = True
        if len(support_not_a_not_y) == 0:
            # for not a -> b: no not a -> not b
            rules.append((((-a,), y), len(support_not_a_y)))
            hash_not_a_y[a] = True
        elif len(support_not_a_y) == 0:
            # for not a -> not b: no not a -> b
            rules.append((((-a,), -y), len(support_not_a_not_y)))
            hash_not_a_not_y[a] = True
    
        n_tests += 1

    
    #print("2k rules: ", rules)
    #for k == 3: generate all a and b -> c rules
    candidates = tuple(combinations(range(1, n_features), 2))
    print("len candidates (k=3):", len(candidates))
    #Test all candidates:
    # a and b => c 
    # a and b => not c 
    # not a and b => c 
    # not a and b => not c 
    for i, candidate in enumerate(candidates):
        
        if ((time.time() - total_time) > time_limit):
            print("time limit madelaine exceeded ...")
            break
        a, b = candidate[0], candidate[1]
        key_a_b = a * b
        intersection_a_b = dict_values_1[a].intersection(dict_values_1[b])
        intersection_not_a_b = dict_values_0[a].intersection(dict_values_1[b])
        intersection_a_not_b = dict_values_1[a].intersection(dict_values_0[b])
        intersection_not_a_not_b=dict_values_0[a].intersection(dict_values_0[b])

        support_a_b_y = intersection_a_b.intersection(dict_values_1[y])
        support_a_b_not_y = intersection_a_b.intersection(dict_values_0[y])
        support_not_a_b_y = intersection_not_a_b.intersection(dict_values_1[y])
        support_not_a_b_not_y = intersection_not_a_b.intersection(dict_values_0[y])
        support_a_not_b_y = intersection_a_not_b.intersection(dict_values_1[y])
        support_a_not_b_not_y = intersection_a_not_b.intersection(dict_values_0[y])
        support_not_a_not_b_y = intersection_not_a_not_b.intersection(dict_values_1[y])
        support_not_a_not_b_not_y = intersection_not_a_not_b.intersection(dict_values_0[y])

        # a and b => c: no a and b => not c 
        if len(support_a_b_not_y) == 0:
            if not (hash_a_y[a] or hash_a_y[b] or hash_a_not_b[key_a_b]): 
                rules.append((((a, b), y), len(support_a_b_y)))
        # a and b => not c: no a and b => c 
        elif len(support_a_b_y) == 0:
            if not (hash_a_not_y[a] or hash_a_not_y[b] or hash_a_not_b[key_a_b]): 
                rules.append((((a, b), -y), len(support_a_b_not_y)))
        
        # not a and b => c: no not a and b => not c 
        if len(support_not_a_b_not_y) == 0:
            if not (hash_not_a_y[a] or hash_a_y[b] or hash_not_a_not_b[key_a_b]): 
                rules.append((((-a, b), y), len(support_not_a_b_y)))
        # not a and b => not c: no not a and b => c 
        elif len(support_not_a_b_y) == 0:
            if not (hash_not_a_not_y[a] or hash_a_not_y[b] or hash_not_a_not_b[key_a_b]): 
                rules.append((((-a, b), -y), len(support_not_a_b_not_y)))
        
        # a and not b => c: no a and not b => not c 
        if len(support_a_not_b_not_y) == 0:
            if not (hash_a_y[a] or hash_not_a_y[b] or hash_a_b[key_a_b]): 
                rules.append((((a, -b), y), len(support_a_not_b_y)))
        # a and not b => not c: no a and not b => c 
        elif len(support_a_not_b_y) == 0:
            if not (hash_a_not_y[a] or hash_not_a_not_y[b] or hash_a_b[key_a_b]): 
                rules.append((((a, -b), -y), len(support_a_not_b_not_y)))

        # not a and not b => c: no not a and not b => not c
        if len(support_not_a_not_b_not_y) == 0:
            if not (hash_not_a_y[a] or hash_not_a_y[b] or hash_not_a_b[key_a_b]): 
                rules.append((((-a, -b), y), len(support_not_a_not_b_y)))
        # not a and not b => -c: no not a and not b => c
        elif len(support_not_a_not_b_y) == 0:
            if not (hash_not_a_not_y[a] or hash_not_a_not_y[b] or hash_not_a_b[key_a_b]):
                rules.append((((-a, -b), -y), len(support_not_a_not_b_not_y)))
        n_tests += 1
    
    print("len(rules):", len(rules))
    print("n_max_rules:", n_max_rules)

    rules = sorted(rules, key=lambda x: x[1], reverse=True)
    rules = [r for r in rules if r[1] != 0]
    if len(rules) > n_max_rules:
        rules = rules[:n_max_rules]
    
    supports = [r[1] for r in rules]
    info_supports = (supports[0], supports[-1], supports[0]/n_instances, supports[-1]/n_instances)
    rules = [r[0] for r in rules]
    len_rules_2 = len(tuple(r for r in rules if len(r[0]) == 1))
    len_rules_3 = len(tuple(r for r in rules if len(r[0]) == 2))

    print("n rules (total):", len(rules))
    
    return info_supports, len_rules_2, len_rules_3, len(rules), (time.time() - total_time), rules

    