import random

import numpy as np
import pyhopper

from ModelTrainer.knnTrainer import train_k_nearest_neighbors
from Utils.constants import META_LEANER_TARGET_COLUMNS
from Utils.datasetHandler import prepared_meta_feature_dataset

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

    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, META_LEANER_TARGET_COLUMNS, target_column)

        search = pyhopper.Search(parameter_group)
        best_params = search.run(
            train_k_nearest_neighbors_warp,
            direction="max",
            steps=number_of_steps,
            # n_jobs="per-gpu"
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
    return np.mean(loses["testing accuracies"])