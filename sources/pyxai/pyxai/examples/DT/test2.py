#fonctions utilisés

from pyxai import Learning, Explainer, Tools
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
# from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
#from decisionNode import DecisionNode
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt

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

# Fonction de transformation des éléments
def transform_feature(feature):
    # Si le feature commence par 'X_', on garde juste le numéro en tant que valeur positive
    if feature.startswith('X_'):
        return int(feature.split('_')[1])
    # Si le feature commence par 'N_', on garde le numéro en tant que valeur négative
    elif feature.startswith('N_'):
        return -int(feature.split('_')[1])
    return feature


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


def convert2(lst):
    def format_element(x):
        return f'X_{x[0]}' if x[0] > 0 else f'N_{-x[0]}'
    
    return [[[format_element(a)], [format_element(b)]] for a, b in lst]

def transform_tuple(tuple_item):
    transformed_tuple = []
    for item in tuple_item:
        parts = item.split(' ', 1)  # Diviser la chaîne en deux parties au premier espace trouvé
        variable = parts[0].replace('X_', '')  # Extraire la variable sans le préfixe 'X_'
        condition = parts[1] if len(parts) > 1 else ''  # Récupérer la condition si elle existe
        transformed_tuple.append((int(variable), condition))  # Ajouter à la liste transformée
    return transformed_tuple

def transform_condition_tuple(condition_list):
    transformed_list = []
    for item in condition_list:
        num, condition = item
        if condition.startswith('<'):
            transformed_list.append(-num)
        else:
            transformed_list.append(num)
    return transformed_list

def compare_taille_tuples(liste1, liste2):
    # Initialiser une liste pour stocker les résultats de la comparaison
    comparisons = []

    # Parcourir les tuples dans les deux listes simultanément
    for tup1, tup2 in zip(liste1, liste2):
        # Calculer la différence de taille entre les tuples et stocker le résultat dans la liste de comparaisons
        difference = abs(len(tup1) - len(tup2))
        comparisons.append(difference)

    # Retourner la liste des différences de taille
    return comparisons
def generelise(rule1, rule2,valeur1,valeur2):
    """
    Check if two rules are in conflict
    """
    # Vérifie si les conditions des règles sont les mêmes
    if valeur1==valeur2:
        if rule1.issubset(rule2):
                return True
    return False


def genereliseclassement(rule1, rule2,valeur1,valeur2):
    """
    Check if two rules are in conflict
    """
    # Vérifie si les conditions des règles sont les mêmes
    # if valeur1.issubset(valeur2):
    if rule1.issubset(rule2):
        return True
    return False


def remove_element_from_key(dictionary, key, element_to_remove):
    """
    Supprime un élément spécifique d'une clé dans un dictionnaire et crée une nouvelle clé avec la valeur associée à la clé d'origine.

    Args:
        dictionary (dict): Le dictionnaire contenant les clés et les valeurs.
        key (frozenset): La clé à modifier.
        element_to_remove: L'élément à supprimer de la clé.

    Returns:
        dict: Le dictionnaire mis à jour.
    """
    # Vérifier si la clé existe dans le dictionnaire
    if key in dictionary:
        # Créer une nouvelle clé sans l'élément à supprimer
        new_key = key - frozenset([element_to_remove])
        # Copier la valeur associée à la clé d'origine
        value = dictionary[key]
        # Supprimer l'ancienne clé du dictionnaire
        del dictionary[key]
        # Ajouter la nouvelle clé avec la même valeur
        dictionary[new_key] = value
    # else:
    #     print("La clé spécifiée n'existe pas dans le dictionnaire.")
    return dictionary


# ######################################################################################################
##########################################################################################################


import pandas as pd
from itertools import combinations
def generate_candidates(itemsets, length,d,target_features,e):
    candidates = set()
    for itemset1 in itemsets:
        for itemset2 in itemsets:
            candidate = itemset1.union(itemset2)
            if len(candidate) == length:
                if length==d:
                    if 'y' in candidate or 'yy' in candidate:
                        candidates.add(candidate)
                else:
                     candidates.add(candidate)
    return candidates

def get_frequent_itemsets(transactions, candidates, min_support):
    itemset_counts = {itemset: 0 for itemset in candidates}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                itemset_counts[candidate] += 1

    num_transactions = len(transactions)
    frequent_itemsets = {itemset for itemset, count in itemset_counts.items() if count / num_transactions >= min_support}
    return frequent_itemsets, {itemset: count / num_transactions for itemset, count in itemset_counts.items() if count / num_transactions >= min_support}

def generate_rules(frequent_itemsets, itemset_supports, min_confidence, rules_to_exclude):
    rules_dict = {}
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
                for subset in map(frozenset, combinations(itemset, len(itemset) - 1)):
                    antecedent = subset
                    consequent = itemset - antecedent
                    # exclude_rule = False  # Flag pour vérifier si une règle doit être exclue
                    # Exclure les règles spécifiques
                    # print('"#######################')
                    # print("antecedent",antecedent)
                    # if (antecedent, consequent) in rules_to_exclude:
                    #     continue
                    # print("final anrtecedant",antecedent)
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

def aprioris(df, min_support, min_confidence,e,d,rules_to_exclude=None):
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
        candidates = generate_candidates(frequent_itemsets, k,d, ['y', 'yy'],e)
        print("okijf")
        print(len(candidates))
        if not candidates:
            break
        frequent_itemsets, supports = get_frequent_itemsets(transactions, candidates, min_support)
        itemset_supports.update(supports)
        all_frequent_itemsets.update(frequent_itemsets)
        k += 1

    rules = generate_rules(all_frequent_itemsets, itemset_supports, min_confidence, rules_to_exclude)
    # print(rules)
    return all_frequent_itemsets, rules
##############################################################################################################

#Explication

glucose = Glucose3()

# I load the dataset
name='/home/cril/Téléchargements/f5/balance-scale_0'
min_support = 0.0005
min_confidence = 1
e=3
d=5
# Diviser le DataFrame en ensembles d'entraînement et de test
dt_learner = Learning.Scikitlearn(name+'.csv', learner_type=Learning.CLASSIFICATION)

# I create a xgboost model: the expert
dt_model = dt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT,seed=0)
# I need the theory.... Currently, we collect clauses related to a binarised instance
# I need an explainer BT
instance, prediction = dt_learner.get_instances(dt_model, n=1)
dt_explainer = Explainer.initialize(dt_model, instance, features_type= name+'.types')
nb_features = len(dt_explainer.binary_representation)
# dt_explainer.add_clause_to_theory((8, -3))
# print("instance:", instance)
# print("binary: ", dt_explainer.binary_representation)
# reason = dt_explainer.minimal_majoritary_reason()
# print("reason: ", reason)
# print("is reason", dt_explainer.is_reason(reason))
# print("#################################################")

# I need to collect the theory related to boolean variables....
# OK
# I collect ALL instances of the dataset
#for clause in bt_model.get_theory(bt_explainer.binary_representation):
    #glucose.add_clause(clause)
binarized = []
#nb_features = len(bt_explainer.binary_representation)
for i, instance in enumerate(dt_learner.data):
    dt_explainer.set_instance(instance)
    binarized.append([0 if l < 0 else 1 for l in dt_explainer.binary_representation] +  [dt_learner.labels[i]])
df=pd.DataFrame(binarized, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
# Convertir toutes les colonnes en type booléen
# df = df.astype(bool)
print("########################")
clauses=[]
print("iciciiii")
# for clause in dt_model.get_theory(dt_explainer.binary_representation):
#     negated_clause = [[-clause[0]],[clause[1]]]
#     clauses.append(negated_clause)
#     glucose.add_clause(clause)
print("1")
print((clauses))
print(dt_explainer.get_theory())

# new_clauses = []
# for clause in clauses:
#   for cl2 in clauses:
#     if clause[1]==cl2[0]:
#       s=[clause[0],cl2[1]]
#       new_clauses.append(s)
# clauses.extend(new_clauses)

print('2')
dt_learner = Learning.Scikitlearn(df, learner_type=Learning.CLASSIFICATION)
# I create a xgboost model: the expert
dt_model = dt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT,seed=4)
ert = Explainer.initialize(dt_model)
# dt_explainer.add_clause_to_theory([3, -4])
# reason = dt_explainer.minimal_majoritary_reason()

feature_names=dt_learner.get_details()[0]['feature_names']

instances_dt_test= dt_learner.get_instances(dt_model, n=None, indexes=Learning.TEST, details=True)
instances_dt_entrainement= dt_learner.get_instances(dt_model, n=None, indexes=Learning.TRAINING, details=True)

X_1=[]
y_1=[]
X_2=[]
y_2=[]
for instance_dict in instances_dt_test:
    instance_dt = instance_dict["instance"]
    label_dt = instance_dict["label"]
    ert.set_instance(instance_dt)
    X_1.append(instance_dt)
    y_1.append(label_dt)
for instance_dict in instances_dt_entrainement:
    instance_dt = instance_dict["instance"]
    label_dt = instance_dict["label"]
    ert.set_instance(instance_dt)
    X_2.append(instance_dt)
    y_2.append(label_dt)
# Création du DataFrame pour X_2
columns_X = [f"X_{i}" for i in range(1, nb_features + 1)]
df_X = pd.DataFrame(X_2, columns=columns_X)

# Création du DataFrame pour y_2
df_y = pd.DataFrame({'y': y_2})

# Concaténation de df_X et df_y
df_entrainement = pd.concat([df_X, df_y], axis=1)

# Affichage du DataFrame final
for i in range(1,df_entrainement.shape[1]):
    df_entrainement[f'N_{i}']=df_entrainement[f'X_{i}'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
df_entrainement['yy'] = df_entrainement['y'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))

print("4")

# trees=[]
# precision_initial=[]
# dt_models = dt_learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT,max_depth=6)
# all_scikit = dt_learner.get_raw_models()
# for i, dt_model in enumerate(dt_models) :
#     # I take scikitLearn model
#     clf = all_scikit[i]
#     tree = clf.tree_
#     # Get the decision tree in tuple form
#     tree_tuple = DecisionNode.parse_decision_tree(tree, feature_names)
#     transformed_tree = DecisionNode.transform_tree(tree_tuple)
#     simplified_tree=DecisionNode.simplify_tree_theorie(transformed_tree, glucose, [])
#     trees.append(simplified_tree)
#     precision_befor_rectify=DecisionNode.precision(simplified_tree, X_1, y_1)
#     precision_initial.append(precision_befor_rectify)

print('5')

print(df_entrainement)
print('##################################################"')
rules_to_exclude=[]
df_filtered = df_entrainement.drop(columns=['y', 'yy'])
frequent_itemsets, rules = aprioris(df_filtered, min_support, min_confidence, e,d,rules_to_exclude)

# Appliquer l'algorithme Apriori pour obtenir les itemsets fréquents
# frequent_itemsets = apriori(df_filtered, min_support=0.3, use_colnames=True)

# Afficher les itemsets fréquents
# print(frequent_itemsets)
rules_list=[]
# Générer les règles d'association à partir des itemsets fréquents
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=1)
print("nbrules",len(rules))
print(rules)
for antecedent, consequent,_ in rules:
    antecedent=convert(antecedent)
    consequent=convert(consequent)
    rules_list.append((antecedent,consequent))
    # if len(consequent) > 1:
    #      for single_consequent in consequent:
    #          rules_list.append((antecedent, (single_consequent,)))  # Ajouter chaque consequent individuellement
    # else:
    #      rules_list.append((antecedent, consequent))  # Si un seul élément, on l'ajoute tel quel



# Afficher le nombre de règles générées
print(f"Nombre de règles: {len(rules_list)}")
# print(rules_list)
rules_list2=[]
for antecedent, consequent in rules_list:
    if len(consequent) > 1:
         for single_consequent in consequent:
             rules_list2.append((antecedent, (single_consequent,)))  # Ajouter chaque consequent individuellement
    else:
         rules_list2.append((antecedent, consequent))  # Si un seul élément, on l'ajoute tel quel

print(f"Nombre de règles: {len(rules_list2)}")
print(rules_list2)
print("###########################################")
theorie2=transform_tuples(rules_list2)
print(theorie2)
print(len(theorie2))
# Initialisation du dictionnaire
# # Initialisation de la liste pour stocker les règles
# rules_list = []

# # Parcourir chaque règle d'association
# for index, rule in rules.iterrows():
#     antecedent = tuple(rule['antecedents'])  # Garder l'antecedent tel quel
#     consequent = tuple(rule['consequents'])  # Extraire le consequent

#     # Si le consequent a plusieurs éléments, les ajouter séparément
#     if len(consequent) > 1:
#         for single_consequent in consequent:
#             rules_list.append((antecedent, (single_consequent,)))  # Ajouter chaque consequent individuellement
#     else:
#         rules_list.append((antecedent, consequent))  # Si un seul élément, on l'ajoute tel quel

# # Appliquer la transformation à la liste des features
# transformed_features = []

# for antecedent, consequent in rules_list:
#     print(antecedent)
#     print(consequent)
#     # Transformer chaque élément des tuples
#     transformed_antecedent = tuple(transform_feature(a) for a in antecedent)
#     transformed_consequent = tuple(transform_feature(c) for c in consequent)
#     # Ajouter le tuple transformé à la nouvelle liste
#     transformed_features.append((transformed_antecedent, transformed_consequent))

# Afficher la liste des features transformées
# print(len(transformed_features))
# print(len(transform_tuples(transformed_features)))
# print("########################################################################## ")
# tyy=[]

# theorie2=transform_tuples(transformed_features)
# print("######################",len(dt_explainer.get_theory()))










print("6")
a=0
b=0
treasean=[]
tsimplify=[]
rt=0
rt1=0
for instance_dict in instances_dt_entrainement[:30]:
    instance_dt = instance_dict["instance"]
    label_dt = instance_dict["label"]
    dt_explainer.set_instance(instance_dt)
    reason = dt_explainer.minimal_majoritary_reason()
    if not(dt_explainer.is_reason(reason)):
        rt+=1
    # print("ko")
    e=dt_explainer.simplify_reason(reason)
    treasean.append(len(reason))
    tsimplify.append(len(e))
    if not(dt_explainer.is_reason(e)):
        print(e)
        print(reason)
        rt1+=1
        dkp1=e
        pnmo1=reason
    # print("jk")
    a+=len(e)
    b+=len(reason)
moreasen1=b/len(instances_dt_entrainement)
moreasensimplify1=a/len(instances_dt_entrainement)

# print(a)
# print(b)



print("##################################################################")
for i in theorie2:
    dt_explainer.add_clause_to_theory(i)
c=0
d=0
print("####################################################################")
treasean1=[]
tsimplify1=[]
i=0
rt2=0
rt3=0
for instance_dict in instances_dt_entrainement[:30]:
    i+=1
    # print(i)
    instance_dt = instance_dict["instance"]
    label_dt = instance_dict["label"]
    dt_explainer.set_instance(instance_dt)
    reason1 = dt_explainer.minimal_majoritary_reason()
    # print("ko")
    if not(dt_explainer.is_reason(reason1)):
        rt2+=1
    e1=dt_explainer.simplify_reason(reason1)
    if not(dt_explainer.is_reason(e1)):
        print(e1)
        print(reason1)
        rt3+=1
        dkp2=e1
        pnmo2=reason1
    treasean1.append(len(reason1))
    tsimplify1.append(len(e1))
    # print("jk")
    d+=len(e1)
    c+=len(reason1)
moreasen2=c/len(instances_dt_entrainement)
moreasensimplify2=d/len(instances_dt_entrainement)
print("sortie")
print("tr1",(treasean))
print("tr2",(treasean1))
print("tsimplify1",(tsimplify))
print("tsimplify2",(tsimplify1))


print("moyenne taille avant d'ajouter les nouvelle clauses sans simplification:",b,reason)
print("moyenne taille avant d'ajouter les nouvelle clauses avec simplification:",a,e)
print("moyenne taille aprés avoir ajouter les nouvelle clauses sans simplification:",c,reason1)
print("moyenne taille aprés avoir ajouter les nouvelle clauses avec simplification:",d,e1)
print("nb rules qui ne sont pas une raison avant d'ajouter les nouvelle clausse et sans simplifaction:",rt)
print("nb rules qui ne sont pas une raison avant d'ajouter les nouvelle clausse et avec simplifaction:",rt1)
print("nb rules qui ne sont pas une raison aprés avoir ajouter les nouvelle clausse et sans simplifaction:",rt2)
print("nb rules qui ne sont pas une raison aprés avoir ajouter les nouvelle clausse et avec simplifaction:",rt3)

