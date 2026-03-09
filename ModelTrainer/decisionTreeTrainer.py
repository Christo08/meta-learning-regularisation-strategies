import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from Utils.constants import MODULE_PATH
from Utils.fileHandler import load_settings

target_columns = ["baseline_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                 "layer_normalisation_testing_loss", "prune_testing_loss", "weight_normalisation_testing_loss" ]

def training_all_decision_trees(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    for target_column in target_columns:
        training_x = np.array(training_set.drop([target_column], axis=1))
        training_y = training_set[target_column]
        testing_x = np.array(testing_set.drop([target_column], axis=1))
        testing_y = testing_set[target_column]
        result = train_decision_tree(settings[target_column],
                                     (training_x, training_y),
                                     (testing_x, testing_y),
                                     seed,
                                     target_column,
                                     kFold)
        result = {
            "model type": "Decision tree",
            "technique": target_column.replace("_testing_loss","").replace("_"," "),
            **result
        }
        results.append(result)
    return results

def train_decision_tree(params, training_set, testing_set, seed, target_column = 'na', kFold = 5):
    kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)
    training_mses = []
    training_accuracy = []
    testing_mses = []
    testing_accuracy = []
    training_x = training_set[0]
    training_y = training_set[1]
    testing_x = testing_set[0]
    testing_y = testing_set[1]
    counter = 1
    for train_idx, test_idx in kf.split(training_x):
        x_train = training_x[train_idx]
        y_train = training_y.iloc[train_idx]
        tree = DecisionTreeClassifier(random_state=seed, **params)
        tree.fit(x_train, y_train)
        y_train_pred = tree.predict(x_train)
        y_test_pred = tree.predict(testing_x)

        training_mses.append(mean_squared_error(y_train, y_train_pred))
        training_accuracy.append(float(np.sum(y_train == y_train_pred)/len(y_train)*100))

        testing_mses.append(mean_squared_error(testing_y, y_test_pred))
        testing_accuracy.append(float(np.sum(testing_y == y_test_pred)/len(y_train)*100))

        if target_column != 'na':
            joblib.dump(tree, f'{MODULE_PATH}DecisionTrees/decision_tree_for_{target_column}_fold_{counter}.pkl')
        counter = counter + 1
    return {
        "training loses": training_mses,
        "training accuracies": training_accuracy,
        "testing loses": testing_mses,
        "testing accuracies": testing_accuracy
    }