import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor


def train_decision_tree(settings, dataset, testing_set, seed, technique, show_info = True, fold=None):
    start_time = time.time()
    if show_info:
        print("")
        print("Model Type: Decision Tree")
        print("Technique: "+technique)
        print("Seed: " + str(seed))

    total_training_mse = 0
    total_validation_mse = 0
    total_testing_mse = 0

    if fold is not None and fold >= 3:
        features = np.array(dataset[0])
        labels = np.array(dataset[1])
        kf = KFold(n_splits=fold, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(kf.split(features)):
            x_training, x_validation = features[train_index], features[test_index]
            y_training, y_validation = labels[train_index], labels[test_index]

            training_mse, validation_mse, testing_mse = training_loop((x_training, y_training),
                                                                      (x_validation, y_validation),
                                                                      testing_set,
                                                                      settings,
                                                                      seed)

            total_training_mse += training_mse
            total_validation_mse += validation_mse
            total_testing_mse += testing_mse

        total_training_mse /= fold
        total_validation_mse /= fold
        total_testing_mse /= fold
    else:
        x_training, x_validation, y_training, y_validation = train_test_split(dataset[0],
                                                                              dataset[1],
                                                                              test_size=0.2,
                                                                              random_state=seed)
        total_training_mse, total_validation_mse, total_testing_mse = training_loop((x_training, y_training),
                                                                                    (x_validation, y_validation),
                                                                                    testing_set,
                                                                                    settings,
                                                                                    seed)

    end_time = time.time()
    duration = end_time - start_time

    result_json_object = {
        "model_type": "Decision Tree",
        "technique": technique,
        "seed": seed,
        "training_MSE": total_training_mse,
        "validation_MSE": total_validation_mse,
        "testing_MSE": total_testing_mse
    }
    return result_json_object, duration

def training_loop(training_set, validation_set, testing_set, settings, seed):
    if settings["has_max_depth"]:
        decision_tree = DecisionTreeRegressor(criterion=settings["criterion"],
                                              splitter=settings["splitter"],
                                              min_samples_split=settings["min_samples_split"],
                                              min_samples_leaf=settings["min_samples_leaf"],
                                              max_depth=settings["max_depth"],
                                              random_state=seed)
    else:
        decision_tree = DecisionTreeRegressor(criterion=settings["criterion"],
                                              splitter=settings["splitter"],
                                              min_samples_split=settings["min_samples_split"],
                                              min_samples_leaf=settings["min_samples_leaf"],
                                              random_state=seed)

    decision_tree = decision_tree.fit(training_set[0], training_set[1])

    training_predict = decision_tree.predict(training_set[0])
    validation_predict = decision_tree.predict(validation_set[0])
    testing_predict = decision_tree.predict(testing_set[0])

    training_mse = mean_squared_error(training_set[1], training_predict)
    validation_mse = mean_squared_error(validation_set[1], validation_predict)
    testing_mse = mean_squared_error(testing_set[1], testing_predict)
    return training_mse, validation_mse, testing_mse