import numpy as np
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier  # Import XGBoost classifier
class DecisionNode:
    def __init__(self, feature_index, left=None, right=None):
        self.feature_index = feature_index
        self.left = left
        self.right = right

    class LeafNode:
        def __init__(self, class_label):
            self.class_label = class_label

    # Here, we use simplification to reduce the number of nodes in the tree, applying the theory mentioned in the paper.
    def simplify_tree_theorie(tree_tuple, glucose, stack):
        def is_node_consistent(node, stack):
            if isinstance(node, tuple):
                feature_index, left, right = node
                stack.append(-feature_index)

                # Check consistency on the left
                left_consistent = glucose.propagate(stack)[0]
                stack.pop()

                # Check consistency on the right
                stack.append(feature_index)
                right_consistent = glucose.propagate(stack)[0]
                stack.pop()
                return left_consistent, right_consistent

            # If it's a leaf, it's still consistent
            return True, True

        def _simplify_tree_theorie(node, stack):
            if isinstance(node, tuple):
                feature_index, left, right = node
                # Check consistency on the left
                left_consistent, right_consistent = is_node_consistent(node, stack)

                if left_consistent:
                    # The left part is consistent, simplify recursively
                    left_simplified = _simplify_tree_theorie(left, stack + [-feature_index])
                else:
                    # The left part is inconsistent, replace with the right
                    left_simplified = _simplify_tree_theorie(right, stack + [feature_index])

                # Reset the tmp list
                tmp = []

                if right_consistent:
                    # The right part is consistent, simplify recursively
                    right_simplified = _simplify_tree_theorie(right, stack + [feature_index])
                else:
                    # The right part is inconsistent, replace with the left
                    right_simplified = _simplify_tree_theorie(left, stack + [-feature_index])

                # If both sides are identical, simplify by replacing with either side
                if str(left_simplified) == str(right_simplified):
                    return left_simplified

                return (feature_index, left_simplified, right_simplified)

            # If it's a leaf, do nothing
            return node

        return _simplify_tree_theorie(tree_tuple, stack)
    

    # Functions for negation of a decision tree.
    def negation_decision_tree(arbre):
        if isinstance(arbre, tuple):
            etiquette, gauche, droite = arbre
            return (etiquette, DecisionNode.negation_decision_tree(gauche), DecisionNode.negation_decision_tree(droite))
        elif arbre == 1:
            return 0
        else:
            return 1
        

    # Functions for conjunction of a decision tree.
    def conjonction_decision_tree(arbre1, arbre2):
        if isinstance(arbre1, tuple) and isinstance(arbre2, tuple):
            etiquette1, gauche1, droite1 = arbre1
            etiquette2, gauche2, droite2 = arbre2
            return (etiquette1, DecisionNode.conjonction_decision_tree(gauche1, arbre2), DecisionNode.conjonction_decision_tree(droite1, arbre2))
        elif arbre1 == 1:
            return arbre2
        else:
            return arbre1
        

    # Functions for disjunction of a decision tree.
    def disjonction_decision_tree(arbre1, arbre2):
        if isinstance(arbre1, tuple) and isinstance(arbre2, tuple):
            etiquette1, gauche1, droite1 = arbre1
            etiquette2, gauche2, droite2 = arbre2
            return (etiquette1, DecisionNode.disjonction_decision_tree(gauche1, arbre2), DecisionNode.disjonction_decision_tree(droite1, arbre2))
        elif arbre1 == 0:
            return arbre2
        else:
            return arbre1
    

    # Function for rectification of a decision tree with a positive decision rule.
    def recttyy(arbreinitial,regle2,glucose):
        arbre2 = DecisionNode.rule_to_treenonty(regle2)
        rect=DecisionNode.disjonction_decision_tree(arbreinitial,arbre2)
        klm=DecisionNode.simplify_tree_theorie(rect, glucose, [])
        return klm
    

    # Function for rectification of a decision tree with a negative decision rule.
    def recttnonyy(arbreinitial,regle1,glucose):
        arbre1 = DecisionNode.rule_to_treety(regle1)
        rect=DecisionNode.conjonction_decision_tree(arbreinitial,arbre1)
        klm=DecisionNode.simplify_tree_theorie(rect, glucose, [])
        return klm
    

    #Transform an decision tree into tuple form with (condition, left child, right child).
    def parse_decision_tree(tree, feature_names, node_index=0):
        # Check if the node is a leaf
        if tree.children_left[node_index] == tree.children_right[node_index]:
            # Get the value of the leaf
            leaf_value = tree.value[node_index].argmax()
            return leaf_value

        # Get the index of the feature used for the split
        feature_index = tree.feature[node_index]
        feature_name = feature_names[feature_index]

        # Splitting condition for the current node
        condition = feature_name

        # Recursion for the right child (if the condition is true)
        right_child_value = DecisionNode.parse_decision_tree(tree, feature_names, tree.children_right[node_index])

        # Recursion for the left child (if the condition is false)
        left_child_value = DecisionNode.parse_decision_tree(tree, feature_names, tree.children_left[node_index])

        if isinstance(right_child_value, int):
            right_child_value = (feature_name, right_child_value)
        if isinstance(left_child_value, int):
            left_child_value = (feature_name, left_child_value)

        return (condition, left_child_value, right_child_value)
    

    #Transforms the condition of a decision tree from (X_i, left_child, right_child) to (i, left_child, right_child).
    def transform_tree(tree):
        # If the tree is a tuple, extract condition and recursively transform left and right children.
        if isinstance(tree, tuple):
            condition, left_child, right_child = tree
            condition = int(condition.split('_')[1]) # Extracting the numeric value from the condition
            left_child = DecisionNode.transform_tree(left_child)
            right_child = DecisionNode.transform_tree(right_child)
            return (condition, left_child, right_child)
        else:
            leaf_value = tree
            if isinstance(leaf_value, str):
                return int(leaf_value.split('_')[1]) # Extracting the numeric value from the leaf
            else:
                return leaf_value
            

    #classifies an instance using a decision tree represented as a tuple.
    def classify(tree, instance):
        if isinstance(tree, tuple):
            # If the current node is a tuple, extract the feature, left, and right children.
            feature, left, right = tree
            #classify the left or right subtree.
            if instance[feature-1] == 0:
                return DecisionNode.classify(left, instance)
            else:
                return DecisionNode.classify(right, instance)
        else:
            # If the current node is a leaf node, return the classification result.
            return tree
        

    #Split into two lists the positive rules and the negative rules that are in the form of tuples.
    def split_list(liste_tuples):
        liste_0 = []
        liste_1 = []
        for tuple_item in liste_tuples:
            valeur = tuple_item[1]
            if valeur == 0:
                liste_0.append(tuple_item)
            elif valeur == 1:
                liste_1.append(tuple_item)
        return liste_0, liste_1
    

    #Transform a positive decision rule into the form of a decision tree with left and right children.
    def rule_to_treety(regle):
        if isinstance(regle, tuple):
            conditions, label = regle
            if len(conditions) == 1:
                if conditions[0]>0:
                    return (conditions[0],1, label)
                elif conditions[0]<0:
                    return (abs(conditions[0]),label,1)

            else:
                condition = conditions[0]
                if condition < 0:
                    condition = abs(condition)
                    return (condition,DecisionNode.rule_to_treety((conditions[1:], label)),1)
                else:
                    return (condition, 1, DecisionNode.rule_to_treety((conditions[1:], label)))
        else:
            return regle
        

    #Transform a negative decision rule into the form of a decision tree with left and right children.
    def rule_to_treenonty(regle):
        if isinstance(regle, tuple):
            conditions, label = regle
            if len(conditions) == 1:
                if conditions[0]>0:
                    return (conditions[0],0,label)
                elif conditions[0]<0:
                    return (abs(conditions[0]),label,0)

            else:
                condition = conditions[0]
                if condition < 0:
                    condition = abs(condition)
                    return (condition,DecisionNode.rule_to_treenonty((conditions[1:], label)),0)
                else:
                    return (condition, 0, DecisionNode.rule_to_treenonty((conditions[1:], label)))
        else:
            return regle
        

    #Calculate the accuracy of a decision tree model by comparing predictions with true labels.
    def precision(vn, X_test1, y_test1):
        s = 0
        predictions = []
        for instance in X_test1:
            predicted_label = DecisionNode.classify(vn, instance)
            predictions.append(predicted_label)
            s += 1

        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

        accuracy = correct_predictions / len(X_test1)
        return accuracy
    def precisionp(vn, X_test1, y_test1):
        # Initialisation
        predictions = []

        # Parcours des instances dans X_test1
        for _, instance in X_test1.iterrows():  # Utilisation de .iterrows() pour obtenir chaque ligne du DataFrame
            # Prédiction de l'étiquette pour chaque instance
            predicted_label = DecisionNode.classify(vn, instance)  # Convertir la ligne en dictionnaire
            predictions.append(predicted_label)

        # Calcul du nombre de prédictions correctes
        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1['y']) if pred == true_label)

        # Calcul de la précision
        accuracy = correct_predictions / len(X_test1)
        return accuracy

    

    #Calculate the number of nodes of a decision tree in tuple form;
    def count_nodes(tree):
        if isinstance(tree, tuple):
            # If the element is a tuple, it's an internal node.
            # Count this node and recursively count nodes in the subtrees.
            count = 1  # Count the current node
            for subtree in tree[1:]:
                count += DecisionNode.count_nodes(subtree)  # Recursively count nodes in the subtrees
            return count
        else:
            # If the element is not a tuple, it's a leaf (0 or 1).
            # Count this leaf (it's a terminal node).
            return 1
        

    #Calculate the depth of a decision tree in tuple form; here, in depth, we also count the root node.
    def tree_depth(tree):
        if not tree:
            return 0  # The tree is empty, so the depth is 0

        # Check if the elements of the tree are indeed tuples
        if not isinstance(tree, tuple):
            return 0  # If the leaf is a number, the depth is 1
        condition, left_child_value, right_child_value = tree

        left_depth= DecisionNode.tree_depth(left_child_value)
        right_depth = DecisionNode.tree_depth(right_child_value)

        # The depth of this tree is the maximum depth between the left subtree and the right subtree, plus 1 for the root
        return max(left_depth, right_depth) + 1
    

    # Calculate the number of correct instances of a decision tree model by comparing predictions with true labels.
    def correct_instance(tree, X_test1, y_test1):
        s = 0
        predictions = []
        for instance in X_test1:
            predicted_label =DecisionNode.classify(tree, instance)
            predictions.append(predicted_label)
            s += 1

        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

        correct = correct_predictions
        return correct
        # Fonction pour vérifier si une instance satisfait les conditions spécifiées par un tuple
    def instance_satisfies_conditions(instance, conditions):
        for condition in conditions:
            col_index = abs(condition) - 1  # L'index de la colonne dans l'instance (0-indexé)
            col_value = 1 if condition > 0 else 0  # Valeur attendue pour cette colonne
            column_name = f'X_{abs(condition)}'
            if instance[column_name] != col_value:
                return False
        return True
    def instance_satisfies_any_conditions(instance, conditions_list):
        satisfied_conditions = []
        for conditions, expected_output in conditions_list:
            if DecisionNode.instance_satisfies_conditions(instance, conditions):
                satisfied_conditions.append((conditions, expected_output))
        return satisfied_conditions
    def instance_satisfies_any_conditions2(instance, conditions_list1):
        satisfied_conditions = []
        for conditions_list in conditions_list1:
            for conditions, expected_output in conditions_list:
                if DecisionNode.instance_satisfies_conditions(instance, conditions):
                    satisfied_conditions.append((conditions, expected_output))
        return satisfied_conditions
    #We are searching for the best hyperparameters for the boosted tree to achieve high accuracy.
    def tuning(dataset):
        def load_dataset(dataset):
            data = pandas.read_csv(dataset).copy()

            # extract labels
            labels = data[data.columns[-1]]
            labels = np.array(labels)

            # remove the label of each instance
            data = data.drop(columns=[data.columns[-1]])

            # extract the feature names
            feature_names = list(data.columns)

            return data.values, labels, feature_names

        X, Y, names = load_dataset(dataset)
        model1 = XGBClassifier()
        param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'n_estimators': [50, 100, 150, 200],
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4],
        #'max_features': ['auto', 'sqrt', 'log2', None],
        #'criterion': ['gini', 'entropy'],
        #'max_leaf_nodes': [None, 10, 20, 30],
        #'min_impurity_decrease': [0.1, 0.2,0.3,0.4],
        #'subsample': [0.8, 0.9, 1.0],
        #'colsample_bytree': [0.8, 0.9, 1.0],
        'learning_rate': [0.02, 0.1,0.2, 0.3],
    }
        gridsearch1 = GridSearchCV(model1,
                                param_grid=param_grid,
                                scoring='balanced_accuracy', refit=True, cv=3,
                                return_train_score=True, verbose=10)

        gridsearch1.fit(X, Y)
        return gridsearch1.best_params_
    
    def tuning2(dataset):
        def load_dataset(dataset):
            data = pandas.read_csv(dataset).copy()

            # extract labels
            labels = data[data.columns[-1]]
            labels = np.array(labels)

            # remove the label of each instance
            data = data.drop(columns=[data.columns[-1]])

            # extract the feature names
            feature_names = list(data.columns)

            return data.values, labels, feature_names

        X, Y, names = load_dataset(dataset)
        model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [3, 4, 5, 6, 8, 9,15],
            
        }
        gridsearch = GridSearchCV(model,
                                param_grid=param_grid,
                                scoring='balanced_accuracy', refit=True, cv=3,
                                return_train_score=True, verbose=10)

        gridsearch.fit(X, Y)
        return gridsearch.best_params_