import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from Utils.constants import *
from Utils.fileHandler import load_settings

def training_all_support_vector_machines(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    for target_column in META_LEANER_TARGET_COLUMNS:
        training_x = np.array(training_set.drop([target_column], axis=1))
        training_y = training_set[target_column]
        testing_x = np.array(testing_set.drop([target_column], axis=1))
        testing_y = testing_set[target_column]
        result = train_support_vector_machines(settings[target_column],
                                               (training_x, training_y),
                                               (testing_x, testing_y),
                                               seed,
                                               target_column,
                                               kFold)
        result = {
            "model type": "svm",
            "technique": target_column.replace("_testing_loss","").replace("_"," "),
            **result
        }
        results.append(result)
    return results

def train_support_vector_machines(params, training_set, testing_set, seed, target_column = 'na', kFold = 5):
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
        y_train = training_y.iloc[train_idx].to_numpy()
        svm_params = params.copy()
        if svm_params.get("kernel") != "poly" and "degree" in svm_params:
            del svm_params["degree"]
        if svm_params.get("kernel") not in ["poly", "sigmoid"] and "coef0" in svm_params:
            del svm_params["coef0"]
        svm = SVC(**svm_params)
        svm.fit(x_train, y_train)
        y_train_pred = svm.predict(x_train)
        y_test_pred = svm.predict(testing_x)

        training_mses.append(mean_squared_error(y_train, y_train_pred))
        training_accuracy.append(accuracy_score(y_train, y_train_pred) * 100)

        testing_y = np.asarray(testing_y)
        testing_mses.append(mean_squared_error(testing_y, y_test_pred))
        testing_accuracy.append(accuracy_score(testing_y, y_test_pred)*100)

        if target_column != 'na':
            joblib.dump(svm, f'{MODULE_PATH}SVM/svm_for_{target_column}_fold_{counter}.pkl')
        counter = counter + 1
    return {
        "training loses": training_mses,
        "training accuracies": training_accuracy,
        "testing loses": testing_mses,
        "testing accuracies": testing_accuracy
    }