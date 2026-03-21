import random

import pandas as pd
import pyhopper

from ModelTrainer.nnTrainer import train_nn
from Utils.constants import PARAMETER_GROUPS, META_LEANER_TARGET_COLUMNS
from Utils.datasetHandler import load_optimiser_dataset, prepared_meta_feature_dataset
from Utils.fileHandler import save_nn_settings, load_settings

MAX_NUMBER_OF_LAYERS = 6
MIN_NUMBER_OF_LAYERS = 2
MAX_NUMBER_OF_EPOCH = 300

basic_parameters = {
    "batch_size": pyhopper.int(16, 256, power_of=2),

    # Stable practical LR range
    "learning_rate": pyhopper.float(0.0005, 0.05, "0.4f"),
    #
    # # Momentum usually works best in high range
    # "momentum": pyhopper.float(0.7, 0.99, "0.4f"),

    # Avoid wasting compute
    "number_of_epochs": pyhopper.int(60, 300, multiple_of=20),

    # Simpler architectures converge more reliably
    "number_of_hidden_layers": pyhopper.int(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS),

    # Prevent huge overparameterized models
    "number_of_neurons_in_layers": pyhopper.int(
        16, 256, multiple_of=16, shape=MAX_NUMBER_OF_LAYERS
    ),
}

dropout_parameters = {
    # Above 0.5 often hurts unless extreme overfitting
    "dropout_layers": pyhopper.float(0.05, 0.5, shape=MAX_NUMBER_OF_LAYERS)
}

prune_parameters = {
    # Large prune amounts destabilize training
    "prune_amount": pyhopper.float(0.05, 0.3, "0.2f"),

    # Less frequent pruning is safer
    "prune_epoch_interval": pyhopper.int(5, 15)
}

weight_decay_parameters = {
    # Typical L2 ranges
    "weight_decay": pyhopper.float(0.00001, 0.15, "0.4f")
}

weight_perturbation_parameters = {
    # Smaller perturbations are usually safer
    "weight_perturbation_amount": pyhopper.float(0.0001, 0.15, "0.f"),

    # Use integer-safe bounds
    "weight_perturbation_interval": pyhopper.int(5, 30)
}

dataset_name = ""
basic_settings = {}
training_set = ""
validation_set = ""
category_columns = []
seed = random.randint(0, 4294967295)


def optimise_nn(dataset_name_input, dataset_settings, parameter_group, basic_settings_parm = None):
    global dataset_name, basic_settings, training_set, validation_set, category_columns

    if parameter_group == PARAMETER_GROUPS[len(PARAMETER_GROUPS) - 1]:
        return True
    dataset_name = dataset_name_input
    sets, category_columns = load_optimiser_dataset(seed, dataset_settings)
    training_set = sets[0]
    validation_set = sets[1]
    if parameter_group == PARAMETER_GROUPS[0]:
        best_params = setup_optimiser_and_run_it(dataset_name, "Basic", basic_parameters, 200)
        path = save_nn_settings(best_params, dataset_name, "")
        basic_settings = load_settings(path)

        best_params = setup_optimiser_and_run_it(dataset_name, "Dropout", dropout_parameters, 50)
        save_nn_settings(best_params, dataset_name, path)

        best_params = setup_optimiser_and_run_it(dataset_name, "Prune", prune_parameters, 100)
        save_nn_settings(best_params, dataset_name, path)

        best_params = setup_optimiser_and_run_it(dataset_name, "Weight_decay", weight_decay_parameters, 50)
        save_nn_settings(best_params, dataset_name, path)

        best_params = setup_optimiser_and_run_it(dataset_name, "Weight_perturbation", weight_perturbation_parameters, 100)
        save_nn_settings(best_params, dataset_name, path)
    else:
        if parameter_group == PARAMETER_GROUPS[1]:
            best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, basic_parameters, 200)
        else:
            basic_settings = basic_settings_parm
            if parameter_group == PARAMETER_GROUPS[2]:
                best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, dropout_parameters, 50)
            elif parameter_group == PARAMETER_GROUPS[3]:
                best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, prune_parameters, 100)
            elif parameter_group == PARAMETER_GROUPS[4]:
                best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, weight_decay_parameters, 50)
            else:
                best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, weight_perturbation_parameters, 100)
        save_nn_settings(best_params, dataset_name, "")
    return parameter_group == PARAMETER_GROUPS[len(PARAMETER_GROUPS) - 1]


def setup_optimiser_and_run_it(dataset_name, parameter_group_name, parameter_group, number_of_steps):
    search = pyhopper.Search(parameter_group)

    best_params = search.run(
        train_nn_warp,
        direction="min",
        steps=number_of_steps,
        checkpoint_path="Data/CheckPoints/NN/" + dataset_name.strip().replace(" ", "_") + "_" + parameter_group_name + "_Checkpoint",
        # n_jobs="per-gpu"
    )
    test_loss = train_nn_warp(best_params)
    print(f"Tuned params for {dataset_name} dataset using {parameter_group_name} parameter group resulting in a of mse: {test_loss}")
    return best_params

def optimise_meta_leaner_nn(dataset):
    global training_set, validation_set, category_columns

    settings = {}

    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, META_LEANER_TARGET_COLUMNS, target_column)
        training_y = training_set[1]
        validation_y = validation_set[1]

        if training_y.shape[1] > validation_y.shape[1]:
            difference = training_y.shape[1] - validation_y.shape[1]
            for i in range(difference):
                validation_y[f"{target_column}_class_{training_y.shape[1]-difference + i}"] = False
        if training_y.shape[1] < validation_y.shape[1]:
            difference = validation_y.shape[1] - training_y.shape[1]
            for i in range(difference):
                training_y[f"{target_column}_class_{validation_y.shape[1]-difference + i}"] = False

        search = pyhopper.Search(basic_parameters)

        training_set = (pd.DataFrame(training_set[0]), training_y)
        validation_set = (pd.DataFrame(validation_set[0]), validation_y)
        best_params = search.run(
            train_nn_warp,
            direction="min",
            steps=150,
            # n_jobs="per-gpu"
        )
        validation_losses = train_nn_warp(best_params)
        print(
        f"Tuned params for random forest for {target_column} resulting in a of mse: {validation_losses}")
        settings[target_column] = best_params
    return settings

def train_nn_warp(params):
    global training_set, validation_set, category_columns
    if "batch_size" in params:
        settings = params
        training_loss_values, training_accuracies_values, testing_loss_values, testing_accuracies_values = train_nn(settings, "", training_set, validation_set, category_columns, seed)
    else:
        settings = {**basic_settings, **params}
        if "dropout_layers" in params:
            training_loss_values, training_accuracies_values, testing_loss_values, testing_accuracies_values = train_nn(settings, "dropout", training_set, validation_set, category_columns, seed)
        elif "prune_amount" in params:
            training_loss_values, training_accuracies_values, testing_loss_values, testing_accuracies_values = train_nn(settings, "prune", training_set, validation_set, category_columns, seed)
        elif "weight_decay" in params:
            training_loss_values, training_accuracies_values, testing_loss_values, testing_accuracies_values = train_nn(settings, "weightDecay", training_set, validation_set, category_columns, seed)
        else:
            training_loss_values, training_accuracies_values, testing_loss_values, testing_accuracies_values= train_nn(settings, "weightPerturbation", training_set, validation_set, category_columns, seed)
    return testing_loss_values