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
        training_result, _, _ = train_meta_support_vector_machines(settings[target_column],
                                                                   cleaned_training_set,
                                                                   cleaned_testing_set,
                                                                   seed,
                                                                   'na',
                                                                   kFold)
        seed = random.randint(0, 4294967295)
        single_training_result, testing_result, path_to_module  = train_meta_support_vector_machines(settings[target_column],
                                                                                                     cleaned_training_set,
                                                                                                     cleaned_testing_set,
                                                                                                     seed,
                                                                                                     target_column,
                                                                                                     0)
        seed = random.randint(0, 4294967295)
        result = {
            "model type": "svm",
            "model path": path_to_module,
            "technique": target_column.replace("_"," "),
            
            "training loses": training_result["training loses"],
            "training accuracies": training_result["training accuracies"],
            "training f1": training_result["training f1"],
            "training true positives": single_training_result["training true positives"][0],
            "training true negatives": single_training_result["training true negatives"][0],
            "training false positives": single_training_result["training false positives"][0],
            "training false negatives": single_training_result["training false negatives"][0],

            "testing loses": testing_result["testing loses"],
            "testing accuracies": testing_result["testing accuracies"],
            "testing f1": testing_result["testing f1"],
            "testing true positives": testing_result["testing true positives"],
            "testing true negatives": testing_result["testing true negatives"],
            "testing false positives": testing_result["testing false positives"],
            "testing false negatives": testing_result["testing false negatives"]
        }
        results.append(result)
    return results

def train_meta_support_vector_machines(params, training_set, testing_set, seed, target_column ='na', kFold = 5):
    training_x = training_set[0]
    training_y = np.argmax(np.asarray(training_set[1]), axis=1)
    testing_x = testing_set[0]
    testing_y = np.argmax(np.asarray(testing_set[1]), axis=1)

    svm_stats = MetaLearnerStats()

    svm_params = params.copy()

    if svm_params.get("kernel") != "poly" and "degree" in svm_params:
        del svm_params["degree"]
    if svm_params.get("kernel") not in ["poly", "sigmoid"] and "coef0" in svm_params:
        del svm_params["coef0"]

    path_to_module = ""
    if target_column != 'na':
        folder_path = f"{MODULE_PATH}SVM\\{datetime.now().strftime("%Y%m%d_%h")}"
        folder_maker(folder_path)

    if kFold == 0:
        svm = SVC(**svm_params, max_iter=100000)
        svm.fit(training_x, training_y)

        y_train_pred = svm.predict(training_x)
        y_test_pred = svm.predict(testing_x)

        svm_stats.update_training_stats(training_y, y_train_pred)
        svm_stats.update_testing_stats(testing_y, y_test_pred)

        if target_column != 'na':
            path_to_module = f'{folder_path}\\{target_column}.pkl'
            joblib.dump(svm, path_to_module)
    else:
        kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)

        counter = 1

        for train_idx, test_idx in kf.split(training_x):
            x_train = training_x[train_idx]
            y_train = training_y[train_idx]

            svm = SVC(**svm_params)
            svm.fit(x_train, y_train)

            y_train_pred = svm.predict(x_train)
            y_test_pred = svm.predict(testing_x)

            svm_stats.update_training_stats(y_train, y_train_pred)
            svm_stats.update_testing_stats(testing_y, y_test_pred)

            if target_column != 'na':
                path_to_module = f'{folder_path}\\{target_column}_fold_{counter}.pkl'
                joblib.dump(svm, path_to_module)

            counter = counter + 1

    return  svm_stats.get_training_stats_json_object(), svm_stats.get_testing_stats_json_object(), path_to_module