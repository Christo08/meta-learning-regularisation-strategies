import random

import numpy as np
import pyhopper

from ModelTrainer.svmTrainer import train_support_vector_machines
from Utils.constants import META_LEANER_TARGET_COLUMNS
from Utils.datasetHandler import prepared_meta_feature_dataset

number_of_steps = 400
parameter_group = {
    "kernel": pyhopper.choice(["linear", "poly", "rbf", "sigmoid"]),
    "C": pyhopper.float(0.1, 100.0, "0.4f"),
    "gamma": pyhopper.choice(["scale", "auto"]),
    "degree": pyhopper.int(2, 5),  # Only used for 'poly' kernel
    "coef0": pyhopper.float(0.0, 1.0, "0.4f")  # Used for 'poly' and 'sigmoid'
}

training_set = {}
validation_set = {}


def optimise_support_vector_machine(dataset):
    global training_set, validation_set

    settings = {}

    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, META_LEANER_TARGET_COLUMNS, target_column)

        search = pyhopper.Search(parameter_group)
        best_params = search.run(
            train_support_vector_machine_warp,
            direction="max",
            steps=number_of_steps,
            # n_jobs="per-gpu"
        )
        validation_losses = train_support_vector_machine_warp(best_params)
        print(
        f"Tuned params for random forest for {target_column} resulting in an accuracy of: {validation_losses}")
        settings[target_column] = best_params
    return settings

def train_support_vector_machine_warp(params):
    global training_set, validation_set
    seed = random.randint(0, 4294967295)
    losses = train_support_vector_machines(params, training_set, validation_set, seed)
    return np.mean(losses["testing accuracies"])