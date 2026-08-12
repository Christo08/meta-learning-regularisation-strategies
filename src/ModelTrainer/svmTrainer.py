import random
from datetime import datetime

import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from src.Utils.constants import *
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import load_settings, folder_maker
from src.Utils.metaLearnerStatsCalculator import MetaLearnerStats


def training_meta_support_vector_machines(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    random.seed(seed)
    for target_column in META_LEANER_TARGET_COLUMNS:
        print(f"Training svm for { target_column.replace("_"," ")}...")
        cleaned_training_set = prepared_meta_feature_dataset(training_set,target_column,False)
        cleaned_testing_set = prepared_meta_feature_dataset(testing_set,target_column,False)
        stats, path_to_module = train_meta_support_vector_machines(settings[target_column],
                                                                   cleaned_training_set,
                                                                   cleaned_testing_set,
                                                                   seed,
                                                                   target_column,
                                                                   kFold)
        training_stats = stats.get_best_training_stats_json_object()
        testing_stats = stats.get_best_testing_stats_json_object()
        result = {
            "model type": "svm",
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

def train_meta_support_vector_machines(params,
                                       training_set,
                                       testing_set,
                                       seed,
                                       target_column ='na',
                                       kFold = 5,
                                       metric_type = OPTIMED_METRIC_OPTIONS[2]):
    training_x = training_set[0]
    training_y = np.argmax(np.asarray(training_set[1]), axis=1)
    testing_x = testing_set[0]
    testing_y = np.argmax(np.asarray(testing_set[1]), axis=1)

    svm_stats = MetaLearnerStats(metric_type)

    svm_params = params.copy()

    if svm_params.get("kernel") != "poly" and "degree" in svm_params:
        del svm_params["degree"]
    if svm_params.get("kernel") not in ["poly", "sigmoid"] and "coef0" in svm_params:
        del svm_params["coef0"]

    if kFold == 0:
        svm = SVC(**svm_params, max_iter=100000)
        svm.fit(training_x, training_y)

        y_train_pred = svm.predict(training_x)
        y_test_pred = svm.predict(testing_x)

        svm_stats.update_training_stats(training_y, y_train_pred)
        svm_stats.update_testing_stats(testing_y, y_test_pred)
        svm_stats.add_module(svm)
    else:
        kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)

        for train_idx, test_idx in kf.split(training_x):
            x_train = training_x[train_idx]
            y_train = training_y[train_idx]

            svm = SVC(**svm_params)
            svm.fit(x_train, y_train)

            y_train_pred = svm.predict(x_train)
            y_test_pred = svm.predict(testing_x)

            svm_stats.update_training_stats(y_train, y_train_pred)
            svm_stats.update_testing_stats(testing_y, y_test_pred)
            svm_stats.add_module(svm)

    path_to_module = ""
    if target_column != 'na':
        folder_path = f"{MODULE_PATH}SVM\\{datetime.now().strftime("%Y%m%d_%h")}"
        folder_maker(folder_path)
        path_to_module = f'{folder_path}\\{target_column}.pkl'
        joblib.dump(svm_stats.get_best_model(), path_to_module)

    return  svm_stats, path_to_module