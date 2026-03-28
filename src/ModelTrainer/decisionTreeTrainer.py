import random
from datetime import datetime

import joblib
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from src.Utils.constants import *
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import load_settings, folder_maker
from src.Utils.metaLearnerStatsCalculator import MetaLearnerStats


def training_meta_decision_trees(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    random.seed(seed)
    for target_column in META_LEANER_TARGET_COLUMNS:
        print(f"Training decision tree for { target_column.replace("_"," ")}...")
        cleaned_training_set = prepared_meta_feature_dataset(training_set,target_column,False)
        cleaned_testing_set = prepared_meta_feature_dataset(testing_set,target_column,False)
        training_result = train_meta_decision_tree(settings[target_column],
                                          cleaned_training_set,
                                          cleaned_testing_set,
                                          seed,
                                          "n.a",
                                          kFold)
        seed = random.randint(0, 4294967295)
        testing_result = train_meta_decision_tree(settings[target_column],
                                                  cleaned_training_set,
                                                  cleaned_testing_set,
                                                  seed,
                                                  target_column,
                                                  0)
        seed = random.randint(0, 4294967295)
        result = {
            "model type": "Decision tree",
            "technique": target_column.replace("_"," "),
            "training loses": training_result["training loses"],
            "training accuracies": training_result["training accuracies"],
            "training f1": training_result["training f1"],
            "testing loses": testing_result["testing loses"],
            "testing accuracies": testing_result["testing accuracies"],
            "testing f1": testing_result["testing f1"],
            "testing true positives": testing_result["testing true positives"],
            "testing true negatives": testing_result["testing true negatives"],
            "testing false positives": testing_result["testing false positives"],
            "testing false negatives": testing_result["testing false negatives"],
        }
        results.append(result)
    return results

def train_meta_decision_tree(params, training_set, testing_set, seed, target_column ='na', kFold = 5):
    training_x = training_set[0]
    training_y = training_set[1].to_numpy()
    testing_x = testing_set[0]
    testing_y = testing_set[1]

    decision_tres_stats = MetaLearnerStats()

    if target_column != 'na':
        folder_path = f"{MODULE_PATH}DecisionTrees\\{datetime.now().strftime("%Y%m%d_%h")}"
        folder_maker(folder_path)

    if kFold == 0:
        tree = DecisionTreeClassifier(random_state=seed, **params)
        tree.fit(training_x, training_y)
        y_train_pred = tree.predict(training_x)
        y_test_pred = tree.predict(testing_x)

        decision_tres_stats.update_stats(training_y, y_train_pred, testing_y, y_test_pred)

        if target_column != 'na':
            joblib.dump(tree, f'{folder_path}\\{target_column}.pkl')
    else:
        kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)

        counter = 1

        for train_idx, test_idx in kf.split(training_x):
            x_train = training_x[train_idx]
            y_train = training_y[train_idx]

            tree = DecisionTreeClassifier(random_state=seed, **params)
            tree.fit(x_train, y_train)

            y_train_pred = tree.predict(x_train)
            y_test_pred = tree.predict(testing_x)

            decision_tres_stats.update_stats(y_train, y_train_pred, testing_y, y_test_pred)

            if target_column != 'na':
                joblib.dump(tree, f'{folder_path}\\{target_column}_fold_{counter}.pkl')

            counter = counter + 1

    return decision_tres_stats.get_stats_json_object()