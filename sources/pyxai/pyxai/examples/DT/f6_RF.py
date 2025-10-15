#fonctions utilisés

from pyxai import Learning, Explainer, Tools ,Builder
from sklearn.model_selection import train_test_split
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
# from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from decisionNode import DecisionNode
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt
def list_to_tuple_pairs(lst):
    if len(lst) % 2 != 0:
        raise ValueError("La liste doit contenir un nombre pair d'éléments.")
    
    return [(lst[i], lst[i+1]) for i in range(0, len(lst), 2)]
def trasforme_list_tuple_to_binaire(tupl,dt_model):
    s=[]
    # print(tupl)
    for k in tupl:
        for n in k:
            # print(n)
            # print(bt_model.map_id_binaries_to_features[abs(n)])
            #s.append(tuple(bt_model.map_id_binaries_to_features[abs(n)]) + (True if n < 0 else False,))
            #s.append(bt_model.map_features_to_id_binaries[n])
            is_inside=False
            for e in dt_model.map_features_to_id_binaries:
                #print(dt_model.map_features_to_id_binaries)
                if e[0]==n: 
                    s.append((dt_model.map_features_to_id_binaries[e])[0])
                    is_inside=True
                elif e[0]==-n:
                    s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                    is_inside=True
            if is_inside is False:
                #print("n not found",n)
                s.append((abs(n),Builder.GT,0.5, True if n < 0 else False))
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
def trasforme_tuple_to_binaire(tupl,dt_model):
    s=[]
    # print(tupl)
    for n in tupl:
        # print(n)
        # print(bt_model.map_id_binaries_to_features[abs(n)])
        #s.append(tuple(bt_model.map_id_binaries_to_features[abs(n)]) + (True if n < 0 else False,))
        #s.append(bt_model.map_features_to_id_binaries[n])
        is_inside=False
        for e in dt_model.map_features_to_id_binaries:
            #print(dt_model.map_features_to_id_binaries)
            if e[0]==n: 
                s.append((dt_model.map_features_to_id_binaries[e])[0])
                is_inside=True
            elif e[0]==-n:
                s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                is_inside=True
        if is_inside is False:
            #print("n not found",n)
            s.append((abs(n),Builder.GT,0.5, True if n < 0 else False))
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

# def generate_candidates(itemsets, length):
#     return set([itemset1.union(itemset2) for itemset1 in itemsets for itemset2 in itemsets if len(itemset1.union(itemset2)) == length])

# def generate_candidates(itemsets, length, target_features):
#     candidates = set()
#     for itemset1 in itemsets:
#         for itemset2 in itemsets:
#             candidate = itemset1.union(itemset2)
#             if len(candidate) == length and any(feature in target_features for feature in candidate):
#                 candidates.add(candidate)
#     return candidates

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
            if 'y' in itemset or 'yy' in itemset:
                for subset in map(frozenset, combinations(itemset, len(itemset) - 1)):
                    antecedent = subset
                    consequent = itemset - antecedent

                    # Exclure les règles spécifiques
                    if (antecedent, consequent) in rules_to_exclude:
                        continue
                    if 'y' in consequent or 'yy' in consequent:
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

def apriori(df, min_support, min_confidence,e,d,rules_to_exclude=None):
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
glucose = Glucose3()
name='balance-scale_0'
data = pd.read_csv(name+'.csv')
name=name
# Dividing the DataFrame into training, testing, and validation sets
train_df, validation_df = train_test_split(data, test_size=0.3, random_state=42)
train_df.columns=data.columns
validation_df.columns=data.columns
# Save the DataFrames to CSV files
train_df.to_csv('train_data.csv', index=False)
validation_df.to_csv('validation_data.csv', index=False)
#best_parameters = DecisionNode.tuning('train_data.csv')
dt_learner = Learning.Scikitlearn('train_data.csv', learner_type=Learning.CLASSIFICATION) # 70%
# I create a xgboost model: the expert
dt_model1 = dt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF,seed=0)
instance, prediction = dt_learner.get_instances(dt_model1, n=1)
# I need an explainer BT
dt_explainer = Explainer.initialize(dt_model1, instance, features_type= name+'.types')
# I need the theory.... Currently, we collect clauses related to a binarised instance
clauses=[]
# I need to collect the theory related to boolean variables....
for clause in dt_model1.get_theory(dt_explainer.binary_representation):
    glucose.add_clause(clause)
    negated_clause = [[-clause[0]],[clause[1]]]
    clauses.append(negated_clause)

binarized_training   = []
raw_validation       = []
label_validation     = []
binarized_validation = []
nb_features = len(dt_explainer.binary_representation)  # nb binarized features
print('uidjhik')
# Iterating through the training set to binarize it
for i, instance in enumerate(dt_learner.data):
    dt_explainer.set_instance(instance)
    binarized_training.append([0 if l < 0 else 1 for l in dt_explainer.binary_representation] +  [dt_learner.labels[i]])
df=pd.DataFrame(binarized_training, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])

# Iterating through the validation set to binarize it
for i, instance in validation_df.iterrows():
    dt_explainer.set_instance(instance[:-1])
    raw_validation.append(instance[:-1])
    label_validation.append(instance[-1])
    binarized_validation.append([0 if l < 0 else 1 for l in dt_explainer.binary_representation] +  [instance[-1]])

training_data=pd.DataFrame(binarized_training, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
training_data.to_csv('training_data.csv', index=False)
dt_learner1 = Learning.Scikitlearn(training_data, learner_type=Learning.CLASSIFICATION)
# We create K fold cross validation models
# Sauvegarder le DataFrame dans un fichier CSV


 #optimisied configuration
# best_parameters2 = DecisionNode.tuning2('train_data.csv')
# dt_models = dt_learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT,seed=0,**best_parameters2)


for i in range(1,df.shape[1]):
    df[f'N_{i}']=df[f'X_{i}'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
df['yy'] = df['y'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))




#default configuration
dt_models = dt_learner1.evaluate(method=Learning.K_FOLDS, output=Learning.RF,seed=0,n_estimators=4)
print("2")
feature_names=dt_learner1.get_details()[0]['feature_names']
all_scikit = dt_learner1.get_raw_models()
# Declare the lists and dictionaries that we will use
trees=[]
number_of_nodes_decision_tree=[]
number_of_nodes_decision_tree0=[]
reasons_with_predictions_dict = {}
reasons_with_predictions_dict1={}  # Dictionary to store the reasons
tuple_of_instance_predictions_boosted_tree=[]
precision_decision_tree_before_correction_on_the_validation_set=[]
X_train1=[]
y_train1=[]
X_test=[]
y_test=[]
X_train_folds=[]
y_train_folds=[]
precision_of_the_bosted_tree_on_the_validation_set=[]
tree_depth=[]
precision_for_each_tree=[]
number_of_different_predictions=[]
print("3")
# Iterating through the 10 decision tree models created with PyXAI
yo=[]
for i, dt_model in enumerate(dt_models) :
    yo.append(dt_model)
    print("dj")
    # I take scikitLearn model
    clf = all_scikit[i]
    # total_nodes = sum(tree.tree_.node_count for tree in clf.estimators_)
    # print(f"Nombre total de nœuds dans tous les arbres : {total_nodes}")
    # number_of_nodes_for_a_single_tree0=clf.tree_.node_count
    # Access the decision tree
    # tree = clf.tree_
    # Get the decision tree in tuple form
    # tree_tuple = DecisionNode.parse_decision_tree(tree, feature_names)
    # transformed_tree = DecisionNode.transform_tree(tree_tuple)
    # Simplify the obtained decision tree
    # simplified_tree=DecisionNode.simplify_tree_theorie(transformed_tree, glucose, [])
    # trees.append(simplified_tree)
    # depth_for_a_single_tree=clf.get_depth()
    # #depth_for_a_single_tree=DecisionNode.tree_depth(transformed_tree)
    # tree_depth.append(depth_for_a_single_tree)
    # number_of_nodes_for_a_single_tree=DecisionNode.count_nodes(transformed_tree)
    # number_of_nodes_decision_tree.append(number_of_nodes_for_a_single_tree0)
    # number_of_nodes_decision_tree0.append(number_of_nodes_for_a_single_tree)
    # I collect  all instances from training set
    instances_dt_training = dt_learner1.get_instances(dt_model, n=None, indexes=Learning.TRAINING, details=True)
    # I collect  all instances from test set
    instances_dt_test= dt_learner1.get_instances(dt_model, n=None, indexes=Learning.TEST, details=True)
    X_train1=[]
    y_train1=[]
    X_1=[]
    y_1=[]
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
    #precision_for_a_single_tree=DecisionNode.precision(transformed_tree, X_train1, y_train1)
    # precision_for_a_single_tree=DecisionNode.precision(transformed_tree, X_1, y_1)
    # precision_for_each_tree.append(precision_for_a_single_tree*100)
    reasons_with_predictions = []
    reasons_with_predictions1=[]
    single_tuple_instance_prediction=[]
    nb = 0
    z=0
    instancemalclasee=[]
    sufficient_reason=[]
    X_test1 = []
    y_test1 = []
    # Store instances and their labels in the lists X_test1 and y_test1 of the validation set
    for id_instance,instance_dict in enumerate(binarized_validation):
        instance_dt = instance_dict[:-1]
        label_dt = instance_dict[-1]
        X_test1.append(instance_dt)
        # prediction_dt=DecisionNode.classify(simplified_tree,instance_dt)
        dt_explainer.set_instance(raw_validation[id_instance])
        y_test1.append(label_dt)
    X_test.append(X_test1) # Store all instances from all decision trees
    y_test.append(y_test1)
    # locf = DecisionNode.precision(simplified_tree, X_test1, y_test1)
    locf = clf.score(X_test1, y_test1)
    precision_decision_tree_before_correction_on_the_validation_set.append(locf)
    print(precision_decision_tree_before_correction_on_the_validation_set)
    # locf1 = clf.score(X_train1, y_train1)
    # print("locf,locf1",locf,locf1)
    # exit(0)
############################################################################################
    

rules_to_exclude=[]
min_support = 0.0005
min_confidence = 1
e=3
d=3
print(df)
# frequent_itemsets, rules = apriori(df, min_support, min_confidence, rules_to_exclude[0])
frequent_itemsets, rules = apriori(df, min_support, min_confidence, e,d,rules_to_exclude)


############################################################################################

# Affichage des itemsets fréquents
# print("Itemsets fréquents:")
# for itemset in frequent_itemsets:
#     print(itemset)

# Affichage des règles d'association
# print("\nRègles d'association:")
# for antecedent, consequent, confidence in rules:
#     print(f"{antecedent} -> {consequent} (confidence: {confidence:.2f})")
print("nb rule",len(rules))
association_dict = {}
antecedents=[]
consequents=[]
# print(rules[0])
for antecedent, consequent,_ in rules:
    antecedent=convert(antecedent)
    consequent=convert(consequent)
    # if antecedent in association_dict:
    #      updated_consequent = consequent.union(association_dict[antecedent])
    #      association_dict[antecedent] = updated_consequent
    # else:
        # association_dict[antecedent] = consequent
    association_dict[antecedent] = consequent
#print(association_dict)
association_dict_copy = dict(association_dict)
new_association_dict = dict(association_dict)
nb_rules=[]
#simplification
# Parcours du dictionnaire
print(len(clauses))
for antecedent, consequent in association_dict.items():
    for clause in clauses:
        if clause[0][0] in list((antecedent)) and clause[1][0] in list(antecedent):
            new_association_dict = remove_element_from_key(new_association_dict, antecedent, clause[1][0])
print("nombre de régles extraire",len(association_dict_copy))
nb_rules.append(len(association_dict_copy))
print("nombre de regles restante aprés simplification",len(new_association_dict))

#généralisation
keys_to_delete = []  # Liste pour stocker les clés à supprimer

for key1 in new_association_dict:
    for key2 in new_association_dict:
        if key1 != key2:
            if generelise((key1), (key2),new_association_dict[key1],new_association_dict[key2]):
                if key2 not in keys_to_delete:
                    keys_to_delete.append(key2)

# Supprimer les clés genéralisé du dictionnaire
for key in keys_to_delete:
    del new_association_dict[key]
# Afficher les règles d'association de classement
print("nb de regles apres généralisation:",len(new_association_dict))
nb_rules.append(len(new_association_dict))
class_association_dict = {}
class_association_dict0={}
for antecedent, consequent in new_association_dict.items():
    if ('y' in consequent):
        class_association_dict[antecedent] = consequent
    if ('yy' in consequent):
        class_association_dict0[antecedent] = consequent

print("nombre regles de classement:",len(class_association_dict)+len(class_association_dict0))
tuple_of_tuples = [(tuple(key), 1) for key in class_association_dict.keys()]
tuple_of_tuples0 = [(tuple(key), 0) for key in class_association_dict0.keys()]




pre=[]
for b in range(len(yo)):
    tree_rectified=yo[b]
    dt_model=yo[b]
    total_nodes=dt_model.n_nodes()
    print(total_nodes)
    ert=Explainer.initialize(dt_model)
    ths=dt_model1.get_theory(dt_explainer.binary_representation)
    print("ths",ths)
    theorie=trasforme_list_tuple_to_binaire(ths,dt_model)
    theorie_clause=ert.condi(conditions=theorie)
    theorie_clause=list_to_tuple_pairs(theorie_clause)
    print("theorie_clause",theorie_clause)
    # # class_association_dict = {frozenset({1}): frozenset({'y'}), frozenset({4,2}): frozenset({1})}
    precisions=[precision_decision_tree_before_correction_on_the_validation_set[b]]
    eft=Explainer.initialize(dt_model)
    for j in tuple_of_tuples:
        i=trasforme_tuple_to_binaire(j[0],dt_model)
        # print("rule",j)
        dt_model = eft.rectify(conditions=i, label=1, tests=False,theory_cnf=theorie_clause)
        precision_tree_rectified=precision(dt_model, X_test[b], y_test[b])
        # tree_rectified = DecisionNode.recttyy(tree_rectified, j,glucose) # Rectify the decision tree with the positive rule j
        # precision_after_rectify=DecisionNode.precision(tree_rectified, X_1, y_1)
        # print("apres rectification",precision_after_rectify)
        precisions.append(precision_tree_rectified)
    for i in tuple_of_tuples0:
        i=trasforme_tuple_to_binaire(i[0],dt_model)
        # print("rule",i)
        dt_model = eft.rectify(conditions=i, label=0, tests=False,theory_cnf=theorie_clause)
        precision_tree_rectified=precision(dt_model, X_test[b], y_test[b])
        # tree_rectified = DecisionNode.recttnonyy(tree_rectified, i,glucose) # Rectify the decision tree with the positive rule j
        # precision_after_rectify=DecisionNode.precision(tree_rectified, X_1, y_1)
        # print("apres rectification",precision_after_rectify)
        precisions.append(precision_tree_rectified)
    pre.append(precisions)
print('pre',pre)
import statistics
median_precision=[]
for i in range(max(len(sublist) for sublist in pre)):
    position_values = [sublist[i] for sublist in pre if i < len(sublist)]
    median_value = statistics.median(position_values) * 100
    median_precision.append(median_value)


print(nb_rules)
print(pre[0])
print((median_precision))
print("avant rectif",median_precision[0])
print("apres rectif",median_precision[nb_rules[1]])
# print(pre)

# # # diagramme en baton
# Liste des noms pour chaque bâton
names = ['ALL', 's']

# Liste des couleurs pour chaque bâton
colors = ['blue','green']  # Assure-toi que cette liste a la même longueur que nb_rules

# Vérification pour s'assurer que les listes ont la même longueur
if len(nb_rules) != len(names) or len(nb_rules) != len(colors):
    raise ValueError("Le nombre de valeurs, de noms et de couleurs doit être le même")

# Création du diagramme à bâtons
plt.figure()
plt.bar(range(len(nb_rules)), nb_rules, color=colors)

# Ajout des noms sur l'axe des x
plt.xticks(range(len(nb_rules)), names)

# Ajout des étiquettes et du titre
plt.xlabel('simplification')
plt.ylabel('rules')
plt.title('nb rules classement')

# Sauvegarde du graphique en tant qu'image
plt.savefig('rules2_'+name)
plt.show()


# Créer des listes pour stocker les médianes pour chaque position
medians_pre = []

# Déterminer le nombre de positions
nb_positions = len(new_association_dict)+1

# Créer une figure
fig, ax = plt.subplots(figsize=(12, 8))

# Itérer sur chaque position
for i in range(nb_positions):
    # Récupérer les données pour 'pre' pour cette position
    data_pre = [sublist[i] for sublist in pre if i < len(sublist)]
    
    # Ajouter les données à la liste des médianes
    medians_pre.append(data_pre)

# Créer un boxplot pour les données 'pre' sans les valeurs aberrantes
boxplots_pre = ax.boxplot(medians_pre, vert=True, patch_artist=True, positions=[p - 0.25 for p in range(nb_positions)], widths=1, showfliers=False)

# Définir la couleur de tous les boxplots 'pre' en bleu clair
for box in boxplots_pre['boxes']:
    box.set_facecolor('lightblue')

# Ajouter les étiquettes des axes et un titre
ax.set_xlabel('Nombre de règles')
ax.set_ylabel('Précision')

# Définir les ticks de l'axe x
ax.set_xticks([p for p in range(0, nb_positions, 5)])
ax.set_xticklabels([f'{i}' for i in range(0, nb_positions, 5)])

# Légende pour distinguer les ensembles de boxplots
ax.legend([boxplots_pre["boxes"][0]], ['Rectification'])

# Définir la couleur de la ligne médiane à noir
for line in boxplots_pre['medians']:
    line.set_color('red')

# Ajuster l'échelle de l'axe x pour augmenter la distance entre les positions
ax.set_xlim(-0.5, nb_positions - 0.5)
plt.savefig('precision2_'+name)
# Afficher le graphique
plt.show()