import random

import numpy as np
import pyhopper
from sklearn.model_selection import train_test_split

from ModelTrainer.randomForestTrainer import train_random_forest
from Utils.datasetHandler import prepared_meta_feature_dataset

number_of_steps = 400
parameter_group = {
    "criterion": pyhopper.choice(["squared_error", "absolute_error", "friedman_mse", "poisson"]),
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
target_columns = ["baseline_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                 "layer_normalisation_testing_loss", "prune_testing_loss", "weight_normalisation_testing_loss" ]


def optimise_random_forest(dataset):
    global training_set, validation_set

    settings = {}

    for target_column in target_columns:
        print(target_column)
        training_set, validation_set = prepared_meta_feature_dataset(dataset, target_columns, target_column)

        search = pyhopper.Search(parameter_group)
        best_params = search.run(
            train_random_forest_warp,
            direction="min",
            steps=number_of_steps,
            # n_jobs="per-gpu"
        )
        validation_losses = train_random_forest_warp(best_params)
        print(
        f"Tuned params for random forest for {target_column} resulting in a of mse: {validation_losses}")
        settings[target_column] = best_params
    return settings

def train_random_forest_warp(params):
    global training_set, validation_set
    seed = random.randint(0, 4294967295)
    losses = train_random_forest(params, training_set, validation_set, seed)
    return np.mean(losses["testing mses"])