import pandas as pd
from itertools import combinations
import time
import random
from mybitset import BitSet
from pyxai.sources.core.tools.encoding import CNF, CNFencoding
# Function to transform the tuples
def rules_to_clauses(rules):
    clauses = []
    for antecedent, consequent in rules:
        # Negation of the first tuple and concatenation with the second
        negated_antecedent = tuple(-x for x in antecedent)
        # Concatenate the transformed first tuple with the second tuple.
        clause = negated_antecedent + (consequent, )
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
    #print("len itemset_counts:", len(itemset_counts))
    #print("Start get_frequent_itemsets")


    for transaction in transactions:
        for i, candidate in enumerate(candidates):
            if candidate.issubset(transaction):
                itemset_counts[i] += 1
    
    num_transactions = len(transactions)
    
    frequent_itemsets = [
        itemset for i, itemset in enumerate(candidates)
        if itemset_counts[i] / num_transactions >= min_support
    ]

    #frequent_itemsets = {
    #    itemset for itemset, count in itemset_counts.items()
    #    if count / num_transactions >= min_support
    #}

    itemset_supports = {
        itemset: itemset_counts[i] / num_transactions
        for i, itemset in enumerate(candidates)
        if itemset_counts[i] / num_transactions >= min_support
    }

    #itemset_supports = {
    #    itemset: count / num_transactions
    #    for itemset, count in itemset_counts.items()
    #    if count / num_transactions >= min_support
    #}
    #print("time get_frequent_itemsets: ", time.time() - st)
    #print("End loop get_frequent_itemsets")
    return frequent_itemsets, itemset_supports

def binary_to_features(explainer, lit):
    tmp = explainer.to_features([lit], details=True)
    return next(iter(tmp))


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


def madelaine(database, *, time_limit=3600, n_max_rules=200000, explainer=None):
    
    #Compute: key -> value
    # dict_values_0: index_feature -> list of indexes of instances where the index_feature value is 0
    # dict_values_1: index_feature -> list of indexes of instances where the index_feature value is 1 

    # dict_hash: multiplication of feature -> list of rules
    total_time = time.time()

    database_tuples = tuple(database.itertuples(index=False, name=None))
    n_instances = len(database_tuples)
    n_features = len(database_tuples[0])
    print("n_instances:", n_instances)
    print("n_features:", n_features)
    dict_values_0 = {i:[] for i in range(1, n_features+1)}
    dict_values_1 = {i:[] for i in range(1, n_features+1)}
    
    hash_x_or_y = [False]*((n_features+1)*n_features+1)
    hash_not_x_or_y = [False]*((n_features+1)*n_features+1)
    hash_x_or_not_y = [False]*((n_features+1)*n_features+1)
    hash_not_x_or_not_y = [False]*((n_features+1)*n_features+1)



    for index_instance, instance in enumerate(database_tuples):
        for index_feature, value in enumerate(instance):
            dict_values_0[index_feature+1].append(index_instance) if value == 0 else dict_values_1[index_feature+1].append(index_instance)

    for i in range(1, n_features+1):
        dict_values_0[i] = set(dict_values_0[i])
        dict_values_1[i] = set(dict_values_1[i])
    
    #for k == 2: generate all a->b rules
    candidates = tuple(combinations(range(1, n_features+1), 2))

    print("len candidates (k=2):", len(candidates))

    #Test all candidates: test a -> b, not(a) -> b, a -> not(b) and not(a) -> not(b)  
    #rules_2 = []
    rules = []

    n_tests = 0
    for candidate in candidates:
        a, b = candidate[0], candidate[1]
        if binary_to_features(explainer, a) == binary_to_features(explainer, b):
            continue
        key = a * b
        support_a_b = dict_values_1[a].intersection(dict_values_1[b])
        support_a_not_b = dict_values_1[a].intersection(dict_values_0[b])
        support_not_a_b = dict_values_0[a].intersection(dict_values_1[b])
        support_not_a_not_b = dict_values_0[a].intersection(dict_values_0[b])


        if len(support_a_not_b) == 0:
            # for a -> b: there is no (a -> not b) in the instances
            rules.append((((a,), b), len(support_a_b)))
            hash_not_x_or_y[key] = True
        elif len(support_a_b) == 0:
            # for a -> not b: no a -> b
            rules.append((((a,), -b), len(support_a_not_b)))
            hash_not_x_or_not_y[key] = True
        if len(support_not_a_not_b) == 0:
            # for not a -> b: no not a -> not b
            rules.append((((-a,), b), len(support_not_a_b)))
            hash_x_or_y[key] = True
        elif len(support_not_a_b) == 0:
            # for not a -> not b: no not a -> b
            rules.append((((-a,), -b), len(support_not_a_not_b)))
            hash_x_or_not_y[key] = True
        #if size_before != len(rules_2):
        #    print("ICI" , binary_to_features(explainer, a), binary_to_features(explainer, b))
        n_tests += 1

    #if len(rules_2) > n_max_rules:
    #    rules_2 = sorted(rules_2, key=lambda x: x[1], reverse=True)[:n_max_rules]
    #rules_2 = [r[0] for r in rules_2]
    #len_rules_2 = len(rules_2)
    #print("n rules (k=2):", len(rules_2))


    #for k == 3: generate all a and b -> c rules
    #random.shuffle(list_combination)
    #Test all candidates:
    # a and b => c 
    # a and b => not c 
    # not a and b => c 
    # not a and b => not c 
    for candidate in combinations(range(1, n_features+1), 3):
        if ((time.time() - total_time) > time_limit):
            break
        a, b, c = candidate[0], candidate[1], candidate[2]  
        
        intersection_a_b = dict_values_1[a].intersection(dict_values_1[b])
        intersection_not_a_b = dict_values_0[a].intersection(dict_values_1[b])
        intersection_a_not_b = dict_values_1[a].intersection(dict_values_0[b])
        intersection_not_a_not_b=dict_values_0[a].intersection(dict_values_0[b])

        key_a_b, key_b_c, key_a_c = a * b, b * c, a * c
        
        support_a_b_c = intersection_a_b.intersection(dict_values_1[c])
        support_a_b_not_c = intersection_a_b.intersection(dict_values_0[c])
        support_not_a_b_c = intersection_not_a_b.intersection(dict_values_1[c])
        support_not_a_b_not_c = intersection_not_a_b.intersection(dict_values_0[c])
        support_a_not_b_c = intersection_a_not_b.intersection(dict_values_1[c])
        support_a_not_b_not_c = intersection_a_not_b.intersection(dict_values_0[c])
        support_not_a_not_b_c = intersection_not_a_not_b.intersection(dict_values_1[c])
        support_not_a_not_b_not_c = intersection_not_a_not_b.intersection(dict_values_0[c])
    

        # a and b => c: no a and b => not c 
        if len(support_a_b_not_c) == 0:
            if not (hash_not_x_or_not_y[key_a_b] or hash_not_x_or_y[key_a_c] or hash_not_x_or_y[key_b_c]): 
                rules.append((((a, b), c), len(support_a_b_c)))
        # a and b => not c: no a and b => c 
        elif len(support_a_b_c) == 0:
            if not (hash_not_x_or_not_y[key_a_b] or hash_not_x_or_not_y[key_a_c] or hash_not_x_or_not_y[key_b_c]): 
                rules.append((((a, b), -c), len(support_a_b_not_c)))
        
        # not a and b => c: no not a and b => not c 
        if len(support_not_a_b_not_c) == 0:
            if not (hash_x_or_not_y[key_a_b] or hash_x_or_y[key_a_c] or hash_not_x_or_y[key_b_c]):
                rules.append((((-a, b), c), len(support_not_a_b_c)))
        # not a and b => not c: no not a and b => c 
        elif len(support_not_a_b_c) == 0:
            if not (hash_x_or_not_y[key_a_b] or hash_x_or_not_y[key_a_c] or hash_not_x_or_not_y[key_b_c]):
                rules.append((((-a, b), -c), len(support_not_a_b_not_c)))
        
        # a and not b => c: no a and not b => not c 
        if len(support_a_not_b_not_c) == 0:
            if not (hash_not_x_or_y[key_a_b] or hash_not_x_or_y[key_a_c] or hash_x_or_y[key_b_c]):
                rules.append((((a, -b), c), len(support_a_not_b_c)))
        # a and not b => not c: no a and not b => c 
        elif len(support_a_not_b_c) == 0:
            if not (hash_not_x_or_y[key_a_b] or hash_not_x_or_not_y[key_a_c] or hash_x_or_not_y[key_b_c]):
                rules.append((((a, -b), -c), len(support_a_not_b_not_c)))

        # not a and not b => c: no not a and not b => not c
        if len(support_not_a_not_b_not_c) == 0:
            if not (hash_x_or_y[key_a_b] or hash_x_or_y[key_a_c] or hash_x_or_y[key_b_c]):
                rules.append((((-a, -b), c), len(support_not_a_not_b_c)))
        # not a and not b => -c: no not a and not b => c
        elif len(support_not_a_not_b_c) == 0:
            if not (hash_x_or_y[key_a_b] or hash_x_or_not_y[key_a_c] or hash_x_or_not_y[key_b_c]):
                rules.append((((-a, -b), -c), len(support_not_a_not_b_not_c)))
        
        n_tests += 1
    
    
    rules = sorted(rules, key=lambda x: x[1], reverse=True)
    if len(rules) > n_max_rules:
        rules = rules[:n_max_rules]

    rules = [r[0] for r in rules] 
    len_rules_2 = len(tuple(r for r in rules if len(r[0]) == 1))
    len_rules_3 = len(tuple(r for r in rules if len(r[0]) == 2))
     
    print("total n tests:", n_tests)
    print("total n rules:", len(rules))

    return len_rules_2, len_rules_3, len(rules), (time.time() - total_time), rules

    

        





def aprioris(df, min_support, min_confidence,max_length,rules_to_exclude=None):
    if rules_to_exclude is None:
        rules_to_exclude = []

    # Convert the rules to exclude into a frozenset to facilitate comparison.
    rules_to_exclude = [(frozenset(antecedent), frozenset(consequent)) for antecedent, consequent in rules_to_exclude]
    size_bitset = len(df.columns) + 1
    transactions = df.apply(lambda row: frozenset(row[row == 1].index), axis=1).tolist()
    transactions_bitset = []
    for transaction in transactions:
        transactions_bitset.append(BitSet(size_bitset, [df.columns.get_loc(element) for element in transaction]))
    transactions = transactions_bitset
    #transactions = df.apply(lambda row: BitSet(size_bitset, row[row == 1].index), axis=1).tolist()


    #candidates = {frozenset([item]) for item in df.columns}
    candidates = [BitSet(size_bitset, [i]) for i in range(len(df.columns))]
    print("candidates:", len(candidates))
    
    print("transactions:", len(transactions_bitset))

    #print("candidates:", candidates)
    
    frequent_itemsets, itemset_supports = get_frequent_itemsets(transactions, candidates, min_support)
    
    print("frequent_itemsets:", len(frequent_itemsets))
    
    all_frequent_itemsets = frequent_itemsets.copy()

    k = 2
    while k<=max_length:
        print("aprioris loop: ", k)
        st = time.time()
        candidates = generate_candidates(frequent_itemsets, k)
        print("candidates:", len(candidates))
        print("time candidates: ", time.time() - st)
        
        if not candidates:
            break
        frequent_itemsets, supports = get_frequent_itemsets(transactions, candidates, min_support)
        print("frequent_itemsets:", len(frequent_itemsets))
        itemset_supports.update(supports)
        all_frequent_itemsets.update(frequent_itemsets)
        k += 1
        print("time loop: ", time.time() - st)
        
    rules = generate_rules(all_frequent_itemsets, itemset_supports, min_confidence)
    return all_frequent_itemsets, rules
