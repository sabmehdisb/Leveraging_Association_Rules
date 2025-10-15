# fonctions utilisés
import json
from pyxai import Learning, Explainer, Tools
import pandas as pd
from pysat.solvers import Glucose3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import aprioris
import time

name = 'compas_0'
data = pd.read_csv(name + '.csv')
train_df, validation_df = train_test_split(data, test_size=0.3, random_state=42)
train_df.columns = data.columns
validation_df.columns = data.columns
# Save the DataFrames to CSV files
train_df.to_csv('train_data.csv', index=False)
validation_df.to_csv('validation_data.csv', index=False)
# Explication

glucose = Glucose3()

# I load the dataset
min_support = 0.1
min_confidence = 1
max_length = 3
d = 5
# Diviser le DataFrame en ensembles d'entraînement et de test
# on doit utiliser aprioris sur tout le dataset sinon on aura un conflit avec la théorie
# dt_learner = Learning.Scikitlearn('train_data.csv', learner_type=Learning.CLASSIFICATION)
dt_learner = Learning.Scikitlearn(name + '.csv', learner_type=Learning.CLASSIFICATION)
# I create a xgboost model: the expert
dt_model = dt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, seed=0)
# I need an explainer BT
instance, prediction = dt_learner.get_instances(dt_model, n=1)
dt_explainer = Explainer.initialize(dt_model, instance, features_type=name + '.types')
nb_features = len(dt_explainer.binary_representation)

binarized = []
raw_validation = []
label_validation = []
binarized_validation = []
# nb_features = len(bt_explainer.binary_representation)
for i, instance in enumerate(dt_learner.data):
    dt_explainer.set_instance(instance)
    binarized.append([0 if l < 0 else 1 for l in dt_explainer.binary_representation] + [dt_learner.labels[i]])
training_data = pd.DataFrame(binarized, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
for i, instance in validation_df.iterrows():
    dt_explainer.set_instance(instance[:-1])
    raw_validation.append(instance[:-1])
    label_validation.append(instance[-1])
    binarized_validation.append([0 if l < 0 else 1 for l in dt_explainer.binary_representation] + [instance[-1]])
validation_data = pd.DataFrame(binarized_validation, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])

# Affichage du DataFrame final
for i in range(1, training_data.shape[1]):
    training_data[f'N_{i}'] = training_data[f'X_{i}'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
training_data['yy'] = training_data['y'].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))

##############################################################
# on doit utiliser aprioris sur tout le dataset sinon on aura un conflit avec la théorie
print(training_data)
print('##################################################')
rules_to_exclude = []
df_filtered = training_data.drop(columns=['y', 'yy'])
start_time = time.time()
# on doit utiliser aprioris sur tout le dataset sinon on aura un conflit avec la théorie
frequent_itemsets, rules = aprioris.aprioris(df_filtered, min_support, min_confidence, max_length, d)
end_time = time.time()
elapsed_time_aprioris = (end_time - start_time)
rules_list = []
# Générer les règles d'association à partir des itemsets fréquents
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=1)
print("nbrules", len(rules))
print(rules[10])
for antecedent, consequent, _ in rules:
    antecedent = aprioris.convert(antecedent)
    consequent = aprioris.convert(consequent)
    rules_list.append((antecedent, consequent))

#############################################################################


# Afficher le nombre de règles générées
print(f"Nombre de règles: {len(rules_list)}")
print(rules_list[10])
# print(rules_list)
rules_list2 = []
for antecedent, consequent in rules_list:
    if len(consequent) > 1:
        for single_consequent in consequent:
            rules_list2.append((antecedent, (single_consequent,)))  # Ajouter chaque consequent individuellement
    else:
        rules_list2.append((antecedent, consequent))  # Si un seul élément, on l'ajoute tel quel

print(f"Nombre de règles: {len(rules_list2)}")
print(rules_list2[10])
print("###########################################")
theorie2 = aprioris.transform_tuples(rules_list2)
print(theorie2[10])
print(len(theorie2))

print("6")
len_reason = 0
treasean = []
nb_is_not_reason = 0
majoritary_reason1 = []
instance1 = []
start_time = time.time()
for id_instance, instance_dict in enumerate(binarized_validation[:100]):
    instance_dt = instance_dict[:-1]
    label_dt = instance_dict[-1]
    # print("instance",instance_dt)
    dt_explainer.set_instance(raw_validation[id_instance])
    # print("instance2",raw_validation[id_instance])
    reason = dt_explainer.majoritary_reason(n_iterations=100)
    instance1.append(raw_validation[id_instance])
    majoritary_reason1.append(reason)
    # print("reasen",reason)
    if not (dt_explainer.is_majoritary_reason(reason)):
        nb_is_not_reason += 1
    # print("ko")
    reason = dt_explainer.to_features(reason)
    treasean.append(len(reason))
    len_reason += len(reason)
moreasen1 = len_reason / len(binarized_validation)
moreasen1 = len_reason / 100
end_time = time.time()
elapsed_time_majoritary_reason1 = (end_time - start_time)

print("teille theorie,", len(dt_explainer.get_theory()))
print('taillethorie2', len(theorie2))
print("##################################################################")
for i in theorie2:
    dt_explainer.add_clause_to_theory(i)

print("####################################################################")

len_reason_theorie2 = 0
nb_is_not_reason2 = 0
treasean1 = []
majoritary_reason2 = []
instance2 = []
start_time = time.time()
for id_instance, instance_dict in enumerate(binarized_validation[:100]):
    instance_dt = instance_dict[:-1]
    label_dt = instance_dict[-1]
    # print("instance",instance_dt)
    dt_explainer.set_instance(raw_validation[id_instance])
    # print("instance2",raw_validation[id_instance])
    reason1 = dt_explainer.majoritary_reason(n_iterations=100)
    instance2.append(raw_validation[id_instance])
    majoritary_reason2.append(reason1)
    # print("reasen",reason1)
    if not (dt_explainer.is_majoritary_reason(reason1)):
        nb_is_not_reason2 += 1
    # print("ko")
    reason1 = dt_explainer.to_features(reason1)
    treasean1.append(len(reason1))
    len_reason_theorie2 += len(reason1)
moreasen2 = len_reason_theorie2 / len(binarized_validation)
moreasen2 = len_reason_theorie2 / 100
end_time = time.time()
elapsed_time_majoritary_reason2 = (end_time - start_time)

count_inf = 0
count_sup = 0
count_eq = 0

# Parcourir les deux listes simultanément
for t, t1 in zip(treasean, treasean1):
    if t1 < t:
        count_inf += 1
    elif t1 > t:
        count_sup += 1
    else:  # t1 == t
        count_eq += 1

# Vérifiez et convertissez en DataFrame si nécessaire
# if isinstance(instance1, list):
#     instance1 = pd.DataFrame(instance1)
# if isinstance(instance2, list):
#     instance2 = pd.DataFrame(instance2)

# # Convertir les DataFrame en une structure JSON-compatible
# instance1 = instance1.to_dict(orient="records")
# instance2 = instance2.to_dict(orient="records")

data_ = {
    "dataset name": name,
    "confidence": min_confidence,
    "support": min_support,
    # "instance1": instance1,
    # "instance2": instance2,
    "majoritary_reason1": majoritary_reason1,
    "majoritary_reason2": majoritary_reason2,
    "comparisons": {
        "inférieurs": count_inf,
        "supérieurs": count_sup,
        "égaux": count_eq
    },
    "sizes": {
        "moyenne taille avant d'ajouter les nouvelles clauses ": moreasen1,
        "moyenne taille après avoir ajouté les nouvelles clauses ": moreasen2
    },
    "rules_not_reasons": {
        "avant ajout des nouvelles clauses ": nb_is_not_reason,
        "après ajout des nouvelles clauses ": nb_is_not_reason2
    },
    "elapsed_time_aprioris": elapsed_time_aprioris
}

# Écriture dans un fichier JSON
file_name = name
with open(file_name + ".json", "w") as file_json:
    json.dump(data_, file_json, indent=4)

# Résultat
print("elapsed_time_majoritary_reason2", elapsed_time_majoritary_reason2)
print("elapsed_time_majoritary_reason1", elapsed_time_majoritary_reason1)
print("elapsed_time_aprioris", elapsed_time_aprioris)
print("Inférieurs :", count_inf)
print("Supérieurs :", count_sup)
print("Égaux :", count_eq)
print('taillethorie2', len(theorie2))
print("teille theorie,", len(dt_explainer.get_theory()))
# print("moyenne taille avant d'ajouter les nouvelle clauses :",len_reason,moreasen1,reason,dt_explainer.to_features(reason))
# print("moyenne taille aprés avoir ajouter les nouvelle clauses :",len_reason_theorie2,moreasen2,reason1,dt_explainer.to_features(reason1))
print("nb rules qui ne sont pas une raison avant d'ajouter les nouvelle clausse :", nb_is_not_reason)
print("nb rules qui ne sont pas une raison aprés avoir ajouter les nouvelle clausse:", nb_is_not_reason2)

import matplotlib.pyplot as plt

# **1er graphique : Comparaison des tailles moyennes des clauses**
categories1 = ["Avant ajout", "Après ajout"]
values1 = [moreasen1, moreasen2]

# Création du graphique
plt.figure(figsize=(8, 6))
plt.bar(categories1, values1, color=["skyblue", "orange"])
plt.title("Comparaison des tailles moyennes des clauses", fontsize=14)
plt.ylabel("Taille moyenne", fontsize=12)
plt.xlabel("Étapes", fontsize=12)
plt.ylim(0, max(values1) + 2)

# Affichage des valeurs sur les barres
for i, v in enumerate(values1):
    plt.text(i, v + 0.1, str(v), ha='center', fontsize=10)

plt.tight_layout()

# Sauvegarde du graphique
plt.savefig(name + "__comparaison_tailles_rules.png")  # Sauvegarder sous forme d'image
plt.show()  # Afficher le graphique

# **2e graphique : Comparaison des valeurs entre catégories**
categories2 = ["Inférieurs", "Supérieurs", "Égaux"]
values2 = [count_inf, count_sup, count_eq]

# Création du graphique
plt.bar(categories2, values2, color=['blue', 'orange', 'green'])
plt.title("Comparaison des valeurs")
plt.ylabel("Nombre d'occurrences")
plt.xlabel("Catégories")

# Sauvegarde du graphique
plt.savefig(name + "__comparaison_nombre_changement.png")  # Sauvegarder sous forme d'image
plt.show()  # Afficher le graphique
