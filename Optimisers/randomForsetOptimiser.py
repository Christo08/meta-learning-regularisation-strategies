import random

import numpy as np
import pyhopper

from ModelTrainer.randomForestTrainer import train_random_forest
from Utils.constants import META_LEANER_TARGET_COLUMNS
from Utils.datasetHandler import prepared_meta_feature_dataset

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


def optimise_random_forest(dataset):
    global training_set, validation_set

    settings = {}

    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, META_LEANER_TARGET_COLUMNS, target_column)

        search = pyhopper.Search(parameter_group)
        best_params = search.run(
            train_random_forest_warp,
            direction="max",
            steps=number_of_steps,
            # n_jobs="per-gpu"
        )
        validation_losses = train_random_forest_warp(best_params)
        print(
        f"Tuned params for random forest for {target_column} resulting in an accuracy of: {validation_losses}")
        settings[target_column] = best_params
    return settings

def train_random_forest_warp(params):
    global training_set, validation_set
    seed = random.randint(0, 4294967295)
    losses = train_random_forest(params, training_set, validation_set, seed)
    return np.mean(losses["testing accuracies"])