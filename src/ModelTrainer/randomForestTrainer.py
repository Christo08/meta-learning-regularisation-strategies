import random
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from src.Utils.constants import *
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import load_settings, folder_maker
from src.Utils.metaLearnerStatsCalculator import MetaLearnerStats


def training_meta_random_forests(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    random.seed(seed)
    for target_column in META_LEANER_TARGET_COLUMNS:
        print(f"Training random forests for { target_column.replace("_"," ")}...")
        cleaned_training_set = prepared_meta_feature_dataset(training_set,target_column,False)
        cleaned_testing_set = prepared_meta_feature_dataset(testing_set,target_column,False)
        stats, path_to_module = train_meta_random_forest(settings[target_column],
                                                         cleaned_training_set,
                                                         cleaned_testing_set,
                                                         seed,
                                                         target_column,
                                                         kFold)
        training_stats = stats.get_best_training_stats_json_object()
        testing_stats = stats.get_best_testing_stats_json_object()
        result = {
            "model type": "Random forest",
            "model path": path_to_module,
            "technique": target_column.replace("_"," "),
            "best fold": stats.get_best_fold(),

            "training loses": training_stats["training loses"],
            "training accuracies": training_stats["training accuracies"],
            "training f1": training_stats["training f1"],
            "training precision": training_stats["training precision"],
            "training true positives": training_stats["training true positives"],
            "training true negatives": training_stats["training true negatives"],
            "training false positives": training_stats["training false positives"],
            "training false negatives": training_stats["training false negatives"],

            "testing loses": testing_stats["testing loses"],
            "testing accuracies": testing_stats["testing accuracies"],
            "testing f1": testing_stats["testing f1"],
            "testing precision": testing_stats["testing precision"],
            "testing true positives": testing_stats["testing true positives"],
            "testing true negatives": testing_stats["testing true negatives"],
            "testing false positives": testing_stats["testing false positives"],
            "testing false negatives": testing_stats["testing false negatives"]
        }
        results.append(result)
    return results

def train_meta_random_forest(params,
                             training_set,
                             testing_set,
                             seed,
                             target_column ='na',
                             kFold = 5,
                             metric_type = OPTIMED_METRIC_OPTIONS[1]):
    training_x = training_set[0]
    training_y = training_set[1].to_numpy()
    testing_x = testing_set[0]
    testing_y = testing_set[1]

    random_forests_stats = MetaLearnerStats(metric_type)

    rf_params = params.copy()
    if rf_params.get("bootstrap") is False:
        rf_params["max_samples"] = None

    if kFold == 0:
        forest = RandomForestClassifier(random_state=seed, **rf_params)
        forest.fit(training_x, training_y)

        y_train_pred = forest.predict(training_x)
        y_test_pred = forest.predict(testing_x)

        random_forests_stats.update_training_stats(training_y, y_train_pred)
        random_forests_stats.update_testing_stats(testing_y, y_test_pred)
        random_forests_stats.add_module(forest)
    else:
        kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)

        for train_idx, test_idx in kf.split(training_x):
            x_train = training_x[train_idx]
            y_train = training_y[train_idx]

            forest = RandomForestClassifier(random_state=seed, **rf_params)
            forest.fit(x_train, y_train)

            y_train_pred = forest.predict(x_train)
            y_test_pred = forest.predict(testing_x)

            random_forests_stats.update_training_stats(y_train, y_train_pred)
            random_forests_stats.update_testing_stats(testing_y, y_test_pred)
            random_forests_stats.add_module(forest)

    path_to_module = ""
    if target_column != 'na':
        folder_path = f"{MODULE_PATH}RandomForest\\{datetime.now().strftime("%Y%m%d_%h")}"
        folder_maker(folder_path)
        path_to_module = f'{folder_path}\\{target_column}.pkl'
        joblib.dump(forest, path_to_module)

    return random_forests_stats, path_to_module