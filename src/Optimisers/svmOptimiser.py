import random
from datetime import datetime

import pyhopper

from src.ModelTrainer.svmTrainer import train_meta_support_vector_machines
from src.Utils.constants import META_LEANER_TARGET_COLUMNS, CHECK_POINTS_PATH, OPTIMED_METRIC_OPTIONS
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import folder_maker

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
selected_metric = ""


def optimise_support_vector_machine(dataset, selected_metrics, direction):
    global training_set, validation_set, selected_metric

    settings = {}
    selected_metric = selected_metrics

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, target_column)

        search = pyhopper.Search(parameter_group)
        check_point_path = f"{CHECK_POINTS_PATH}Meta-learners\\SVM"
        folder_maker(check_point_path)
        best_params = search.run(
            train_support_vector_machine_warp,
            direction=direction,
            steps=number_of_steps,
            checkpoint_path=f"{check_point_path}\\{target_column}_{timestamp}"
        )
        validation_loses = train_support_vector_machine_warp(best_params)
        print(f"Tuned params for svm for {target_column} resulting in {validation_loses} {selected_metrics}")
        settings[target_column] = best_params
    return settings

def train_support_vector_machine_warp(params):
    global training_set, validation_set, selected_metric
    seed = random.randint(0, 4294967295)
    loses = train_meta_support_vector_machines(params, training_set, validation_set, seed,kFold=0)
    if selected_metric == OPTIMED_METRIC_OPTIONS[0]:
        return loses["testing accuracies"]
    elif selected_metric == OPTIMED_METRIC_OPTIONS[1]:
        return loses["testing f1"]
    elif selected_metric == OPTIMED_METRIC_OPTIONS[2]:
        return loses["testing loses"]
    else:
        return loses["testing true positives"]/(loses["testing true positives"]+loses["testing false positives"])