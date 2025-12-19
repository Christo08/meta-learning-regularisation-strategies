import random

import pyhopper

from Utils.fileHandler import save_settings, load_settings
from Utils.menus import show_menu
from ModelTrainer.nnTrainer import train_nn
from Utils.datasetHandler import load_optimiser_dataset

max_number_of_layers = 10
min_number_of_layers = 5
max_number_of_epoch = 1000

parameter_groups = ["All", "Basic", "Dropout", "Prune", "Weight decay", "Weight perturbation", "Back"]

basic_parameters = {
    "batch_size": pyhopper.int(16, 1024, power_of=2),
    "learning_rate":  pyhopper.float(0.0001,0.5,"0.4f"),
    "momentum": pyhopper.float(0.0001,0.5,"0.4f"),
    "number_of_epochs": pyhopper.int(50, max_number_of_epoch, multiple_of=20),
    "number_of_hidden_layers": pyhopper.int(min_number_of_layers, max_number_of_layers),
    "number_of_neurons_in_layers": pyhopper.int(5, 1000, multiple_of=5, shape=max_number_of_layers)
}

dropout_parameters = {
    "dropout_layers": pyhopper.float(0.01, 0.75, shape=max_number_of_layers)
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
    "weight_perturbation_interval": pyhopper.int(max_number_of_epoch / 250, max_number_of_epoch / 10)
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
            path = save_settings(best_params, dataset_name, "")
            basic_settings = load_settings(path)

            best_params = setup_optimiser_and_run_it(dataset_name, "Dropout", dropout_parameters, 50)
            save_settings(best_params, dataset_name, path)

            best_params = setup_optimiser_and_run_it(dataset_name, "Prune", prune_parameters, 100)
            save_settings(best_params, dataset_name, path)

            best_params = setup_optimiser_and_run_it(dataset_name, "Weight_decay", weight_decay_parameters, 50)
            save_settings(best_params, dataset_name, path)

            best_params = setup_optimiser_and_run_it(dataset_name, "Weight_perturbation", weight_perturbation_parameters, 100)
            save_settings(best_params, dataset_name, path)
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
            save_settings(best_params, dataset_name, "")
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
    print(f"Tuned params for {dataset_name} dataset using {parameter_group_name} parameter group resulting in a of loss: {test_loss}")
    return best_params


def train_nn_warp(params):
    global training_set, validation_set, category_columns
    if "batch_size" in params:
        settings = params
        training_losses, validation_losses = train_nn(settings, "", training_set, validation_set, category_columns, seed)
    else:
        settings = {**basic_settings, **params}
        if "dropout_layers" in params:
            training_losses, validation_losses = train_nn(settings, "dropout", training_set, validation_set, category_columns, seed)
        elif "prune_amount" in params:
            training_losses, validation_losses = train_nn(settings, "prune", training_set, validation_set, category_columns, seed)
        elif "weight_decay" in params:
            training_losses, validation_losses = train_nn(settings, "weightDecay", training_set, validation_set, category_columns, seed)
        else:
            training_losses, validation_losses = train_nn(settings, "weightPerturbation", training_set, validation_set, category_columns, seed)
    return validation_losses