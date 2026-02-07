import random

import numpy as np
import pandas as pd
import pyhopper
from sklearn.model_selection import train_test_split

from Utils.fileHandler import save_nn_settings, load_settings
from Utils.menus import show_menu
from ModelTrainer.nnTrainer import train_nn
from Utils.datasetHandler import load_optimiser_dataset, apply_one_hot_encode, prepared_meta_feature_dataset

MAX_NUMBER_OF_LAYERS = 10
MIN_NUMBER_OF_LAYERS = 2
MAX_NUMBER_OF_EPOCH = 1000

parameter_groups = ["All", "Basic", "Dropout", "Prune", "Weight decay", "Weight perturbation", "Back"]
meta_learning_target_columns = ["baseline_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                                "layer_normalisation_testing_loss", "prune_testing_loss", "weight_normalisation_testing_loss" ]

basic_parameters = {
    "batch_size": pyhopper.int(16, 1024, power_of=2),
    "learning_rate":  pyhopper.float(0.0001,0.5,"0.4f"),
    "momentum": pyhopper.float(0.0001,0.5,"0.4f"),
    "number_of_epochs": pyhopper.int(50, MAX_NUMBER_OF_EPOCH, multiple_of=20),
    "number_of_hidden_layers": pyhopper.int(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS),
    "number_of_neurons_in_layers": pyhopper.int(5, 1000, multiple_of=5, shape=MAX_NUMBER_OF_LAYERS)
}

dropout_parameters = {
    "dropout_layers": pyhopper.float(0.01, 0.75, shape=MAX_NUMBER_OF_LAYERS)
}

prune_parameters = {
    "prune_amount": pyhopper.float(0.01, 0.75, "0.2f"),
    "prune_epoch_interval": pyhopper.int(4, 20)
}

weight_decay_parameters = {
    "weight_decay": pyhopper.float(0.01, 0.5, "0.2f")
}

weight_perturbation_parameters = {
    "weight_perturbation_amount": pyhopper.float(0.01, 1.00, "0.2f"),
    "weight_perturbation_interval": pyhopper.int(MAX_NUMBER_OF_EPOCH / 250, MAX_NUMBER_OF_EPOCH / 10)
}

dataset_name = ""
basic_settings = {}
training_set = ""
validation_set = ""
category_columns = []
seed = random.randint(0, 4294967295)


def optimise_nn(dataset_name_input, dataset_settings):
    global dataset_name, basic_settings, training_set, validation_set, category_columns
    parameter_group = show_menu("Select parameter group by entering a number:", parameter_groups)
    if parameter_group == parameter_groups[len(parameter_groups) - 1]:
        return True
    dataset_name = dataset_name_input
    sets, category_columns = load_optimiser_dataset(seed, dataset_settings)
    training_set = sets[0]
    validation_set = sets[1]
    while parameter_group != parameter_groups[len(parameter_groups) - 1]:
        if parameter_group == parameter_groups[0]:
            best_params = setup_optimiser_and_run_it(dataset_name, "Basic", basic_parameters, 300)
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
            if parameter_group == parameter_groups[1]:
                best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, basic_parameters, 300)
            else:
                basic_settings = load_settings(input("Enter the path to the basic settings file of the NN:"))
                if parameter_group == parameter_groups[2]:
                    best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, dropout_parameters, 50)
                elif parameter_group == parameter_groups[3]:
                    best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, prune_parameters, 100)
                elif parameter_group == parameter_groups[4]:
                    best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, weight_decay_parameters, 50)
                else:
                    best_params = setup_optimiser_and_run_it(dataset_name, parameter_group, weight_perturbation_parameters, 100)
            save_nn_settings(best_params, dataset_name, "")
        parameter_group = show_menu("Select parameter group by entering a number:", parameter_groups)
    return parameter_group == parameter_groups[len(parameter_groups) - 1]


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

    for target_column in meta_learning_target_columns:
        training_set, validation_set = prepared_meta_feature_dataset(dataset, meta_learning_target_columns, target_column)
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