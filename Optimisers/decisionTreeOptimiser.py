import random

import numpy as np
import pyhopper

from ModelTrainer.decisionTreeTrainer import train_decision_tree
from Utils.constants import META_LEANER_TARGET_COLUMNS
from Utils.datasetHandler import prepared_meta_feature_dataset

number_of_steps = 400
parameter_group = {
    "criterion": pyhopper.choice(["gini", "entropy", "log_loss"]),
    "splitter": pyhopper.choice(["best", "random"]),
    "max_depth": pyhopper.int(1, 200),
    "min_samples_split": pyhopper.int(2, 60),
    "min_samples_leaf": pyhopper.int(1, 60),
    "ccp_alpha": pyhopper.float(0.0, 0.5, "0.4f")
}
training_set = {}
validation_set = {}


def optimise_decision_tree(dataset):
    global training_set, validation_set

    settings = {}

    for target_column in META_LEANER_TARGET_COLUMNS:
        print(target_column)
        training_set, validation_set = prepared_meta_feature_dataset(dataset, META_LEANER_TARGET_COLUMNS, target_column)

        search = pyhopper.Search(parameter_group)
        best_params = search.run(
            train_decision_tree_warp,
            direction="max",
            steps=number_of_steps,
            # n_jobs="per-gpu"
        )
        validation_losses = train_decision_tree_warp(best_params)
        print(
        f"Tuned params for decision tree for {target_column} resulting in a of mse: {validation_losses}")
        settings[target_column] = best_params
    return settings

def train_decision_tree_warp(params):
    global training_set, validation_set
    seed = random.randint(0, 4294967295)
    losses = train_decision_tree(params, training_set, validation_set, seed)
    return np.mean(losses["testing accuracies"])