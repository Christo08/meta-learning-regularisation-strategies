import random
from datetime import datetime

import numpy as np
import pyhopper

from src.ModelTrainer.randomForestTrainer import train_random_forest
from src.Utils.constants import META_LEANER_TARGET_COLUMNS, CHECK_POINTS_PATH, OPTIMED_METRIC_OPTIONS
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import folder_maker

number_of_steps = 400
parameter_group = {
    "criterion": pyhopper.choice(["gini", "entropy", "log_loss"]),
    "max_depth": pyhopper.int(1, 400),
    "min_samples_split": pyhopper.int(2, 60),
    "min_samples_leaf": pyhopper.int(1, 60),
    "ccp_alpha": pyhopper.float(0.0, 0.5, "0.4f"),
    "n_estimators": pyhopper.int(10, 300),
    "bootstrap": pyhopper.choice([True, False]),
    "max_samples": pyhopper.float(0.1, 1.0, "0.2f")
}
training_set = {}
validation_set = {}
selected_metric = ""


def optimise_random_forest(dataset, selected_metrics, direction):
    global training_set, validation_set, selected_metric

    settings = {}
    selected_metric = selected_metrics

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, META_LEANER_TARGET_COLUMNS, target_column)

        search = pyhopper.Search(parameter_group)
        check_point_path = f"{CHECK_POINTS_PATH}Meta-learners\\RandomForest"
        folder_maker(check_point_path)
        best_params = search.run(
            train_random_forest_warp,
            direction=direction,
            steps=number_of_steps,
            checkpoint_path=f"{check_point_path}\\{target_column}_{timestamp}"
        )
        validation_loses = train_random_forest_warp(best_params)
        print(
        f"Tuned params for random forest for {target_column} resulting in an accuracy of: {validation_loses}")
        settings[target_column] = best_params
    return settings

def train_random_forest_warp(params):
    global training_set, validation_set, selected_metric
    seed = random.randint(0, 4294967295)
    loses = train_random_forest(params, training_set, validation_set, seed)
    if selected_metric == OPTIMED_METRIC_OPTIONS[0]:
        return np.mean(loses["testing accuracies"])
    elif selected_metric == OPTIMED_METRIC_OPTIONS[1]:
        return np.mean(loses["testing f1"])
    elif selected_metric == OPTIMED_METRIC_OPTIONS[2]:
        return np.mean(loses["testing loses"])
    else:
        return np.mean(loses["testing true positives"])