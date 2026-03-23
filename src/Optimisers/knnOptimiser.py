import random
from datetime import datetime

import numpy as np
import pyhopper

from src.ModelTrainer.knnTrainer import train_k_nearest_neighbors
from src.Utils.constants import META_LEANER_TARGET_COLUMNS, CHECK_POINTS_PATH
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


def optimise_k_nearest_neighbors(dataset):
    global training_set, validation_set

    settings = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, META_LEANER_TARGET_COLUMNS, target_column)

        search = pyhopper.Search(parameter_group)
        check_point_path = f"{CHECK_POINTS_PATH}Meta-learners\\KNN"
        folder_maker(check_point_path)
        best_params = search.run(
            train_k_nearest_neighbors_warp,
            direction="max",
            steps=number_of_steps,
            checkpoint_path=f"{check_point_path}\\{target_column}_{timestamp}"
        )
        validation_loses = train_k_nearest_neighbors_warp(best_params)
        print(
        f"Tuned params for KNearestNeighbors for {target_column} resulting in an accuracy of: {validation_loses}")
        settings[target_column] = best_params
    return settings

def train_k_nearest_neighbors_warp(params):
    global training_set, validation_set
    seed = random.randint(0, 4294967295)
    loses = train_k_nearest_neighbors(params, training_set, validation_set, seed)
    return np.mean(loses["testing f1"])