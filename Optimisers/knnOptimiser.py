import random

import numpy as np
import pyhopper
from sklearn.model_selection import train_test_split

from ModelTrainer.knnTrainer import train_k_nearest_neighbors
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
target_columns = ["baseline_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                 "layer_normalisation_testing_loss", "prune_testing_loss", "weight_normalisation_testing_loss" ]


def optimise_k_nearest_neighbors(dataset):
    global training_set, validation_set

    settings = {}

    for target_column in target_columns:
        print(target_column)
        training_set, validation_set = prepared_meta_feature_dataset(dataset, target_columns, target_column)

        search = pyhopper.Search(parameter_group)
        best_params = search.run(
            train_k_nearest_neighbors_warp,
            direction="min",
            steps=number_of_steps,
            # n_jobs="per-gpu"
        )
        validation_losses = train_k_nearest_neighbors_warp(best_params)
        print(
        f"Tuned params for KNearestNeighbors for {target_column} resulting in a of mse: {validation_losses}")
        settings[target_column] = best_params
    return settings

def train_k_nearest_neighbors_warp(params):
    global training_set, validation_set
    seed = random.randint(0, 4294967295)
    losses = train_k_nearest_neighbors(params, training_set, validation_set, seed)
    return np.mean(losses["testing mses"])