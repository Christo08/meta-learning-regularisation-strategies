from datetime import datetime

import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from src.Utils.constants import *
from src.Utils.fileHandler import load_settings, folder_maker
from src.Utils.metaLearnerStatsCalculator import MetaLearnerStats


def training_meta_decision_trees(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    for target_column in META_LEANER_TARGET_COLUMNS:
        print(f"Training decision tree for { target_column.replace("_"," ")}...")
        training_x = np.array(training_set.drop([target_column], axis=1))
        training_y = training_set[target_column]
        testing_x = np.array(testing_set.drop([target_column], axis=1))
        testing_y = testing_set[target_column]
        result = train_meta_decision_tree(settings[target_column],
                                          (training_x, training_y),
                                          (testing_x, testing_y),
                                          seed,
                                          target_column,
                                          kFold)
        result = {
            "model type": "Decision tree",
            "technique": target_column.replace("_"," "),
            **result
        }
        results.append(result)
    return results

def train_meta_decision_tree(params, training_set, testing_set, seed, target_column ='na', kFold = 5):
    kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)

    decision_tres_stats = MetaLearnerStats()

    training_x = training_set[0]
    training_y = training_set[1]
    testing_x = testing_set[0]
    testing_y = testing_set[1]
    counter = 1
    for train_idx, test_idx in kf.split(training_x):
        x_train = training_x[train_idx]
        y_train = training_y.iloc[train_idx].to_numpy()
        tree = DecisionTreeClassifier(random_state=seed, **params)
        tree.fit(x_train, y_train)
        y_train_pred = tree.predict(x_train)
        y_test_pred = tree.predict(testing_x)

        decision_tres_stats.update_stats(y_train, y_train_pred, testing_y, y_test_pred)

        if target_column != 'na':
            folder_path = f"{MODULE_PATH}DecisionTrees\\{datetime.now().strftime("%Y%m%d_%h")}"
            folder_maker(folder_path)
            joblib.dump(tree, f'{folder_path}\\decision_tree_for_{target_column}_fold_{counter}.pkl')
        counter = counter + 1
    return decision_tres_stats.get_stats_json_object()