import random
from datetime import datetime

import numpy as np
import pyhopper

from src.ModelTrainer.knnTrainer import train_meta_k_nearest_neighbors
from src.Utils.constants import META_LEANER_TARGET_COLUMNS, CHECK_POINTS_PATH, OPTIMED_METRIC_OPTIONS
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import folder_maker

number_of_steps = 400
parameter_group = {
    "n_neighbors": pyhopper.int(1, 50),
    "weights": pyhopper.choice(["uniform", "distance"]),
    "algorithm": pyhopper.choice(["auto", "ball_tree", "kd_tree", "brute"]),
    "leaf_size": pyhopper.int(10, 100),
    "p": pyhopper.int(1, 5),
    "metric": pyhopper.choice(["minkowski", "euclidean", "manhattan", "chebyshev"])
}
training_set = {}
validation_set = {}
metric_type = ""


def optimise_k_nearest_neighbors(training_dataset, validation_dataset, selected_metric_type, direction):
    global training_set, validation_set, metric_type

    settings = {}
    metric_type = selected_metric_type

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set = prepared_meta_feature_dataset(training_dataset, target_column, False)
        validation_set = prepared_meta_feature_dataset(validation_dataset, target_column, False)

        search = pyhopper.Search(parameter_group)
        check_point_path = f"{CHECK_POINTS_PATH}Meta-learners\\KNN"
        folder_maker(check_point_path)
        best_params = search.run(
            train_k_nearest_neighbors_warp,
            direction=direction,
            steps=number_of_steps,
            checkpoint_path=f"{check_point_path}\\{target_column}_{timestamp}"
        )
        validation_loses = train_k_nearest_neighbors_warp(best_params)
        print(f"Tuned params for knn for {target_column} resulting in {validation_loses} {metric_type}")
        settings[target_column] = best_params
    return settings

def train_k_nearest_neighbors_warp(params):
    global training_set, validation_set, metric_type
    seed = random.randint(0, 4294967295)
    stats, _ = train_meta_k_nearest_neighbors(params, training_set, validation_set, seed, kFold=5, metric_type=metric_type)
    testing_stats = stats.get_testing_stats_json_object()
    if metric_type == OPTIMED_METRIC_OPTIONS[0]:
        return np.mean(testing_stats["testing accuracies"])
    elif metric_type == OPTIMED_METRIC_OPTIONS[1]:
        return np.mean(testing_stats["testing f1"])
    elif metric_type == OPTIMED_METRIC_OPTIONS[2]:
        return np.mean(testing_stats["testing loses"])
    else:
        return np.mean(testing_stats["testing precision"])