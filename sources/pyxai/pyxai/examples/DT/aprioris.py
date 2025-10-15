import pandas as pd
from itertools import combinations
# Fonction pour transformer les tuples
def transform_tuples(tuples):
    transformed_list = []
    
    for antecedent, consequent in tuples:
        # Négatif du premier tuple et concaténation avec le second
        negated_antecedent = tuple(-x for x in antecedent)
        # Concaténer le premier tuple transformé avec le second tuple
        combined_tuple = negated_antecedent + tuple(consequent)
        transformed_list.append(combined_tuple)
    
    return transformed_list



#convertir les collones en numéros
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
    # Générer des candidats en combinant des itemsets fréquents
    return {
        frozenset(itemset1.union(itemset2))
        for itemset1 in itemsets
        for itemset2 in itemsets
        if len(itemset1.union(itemset2)) == length
    }


def get_frequent_itemsets(transactions, candidates, min_support):
    # Compter les occurrences des candidats dans les transactions
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


def generate_rules(frequent_itemsets, itemset_supports, min_confidence):
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

    # Générer les règles seulement si la combinaison existe dans itemset_supports
    rules = [(antecedent, consequent, itemset_supports[antecedent | consequent] / itemset_supports[antecedent])
             for antecedent, consequent in rules_dict.items() if antecedent | consequent in itemset_supports]
    return rules

def aprioris(df, min_support, min_confidence,e,rules_to_exclude=None):
    if rules_to_exclude is None:
        rules_to_exclude = []

    # Convertir les règles à exclure en frozenset pour faciliter la comparaison
    rules_to_exclude = [(frozenset(antecedent), frozenset(consequent)) for antecedent, consequent in rules_to_exclude]
    transactions = df.apply(lambda row: frozenset(row[row == 1].index), axis=1).tolist()

    candidates = {frozenset([item]) for item in df.columns}
    frequent_itemsets, itemset_supports = get_frequent_itemsets(transactions, candidates, min_support)
    all_frequent_itemsets = frequent_itemsets.copy()

    k = 2
    while k<=e:
        # candidates = generate_candidates(frequent_itemsets, k)
        candidates = generate_candidates(frequent_itemsets, k)
        print("okijf")
        print(len(candidates))
        if not candidates:
            break
        frequent_itemsets, supports = get_frequent_itemsets(transactions, candidates, min_support)
        itemset_supports.update(supports)
        all_frequent_itemsets.update(frequent_itemsets)
        k += 1

    rules = generate_rules(all_frequent_itemsets, itemset_supports, min_confidence)
    # print(rules)
    return all_frequent_itemsets, rules
