import json
import os

from Utils.fileHandler import save_nn_settings


def create_generic_nn_setting():
    settings = []
    generic_nn_settings ={
        "batch_size": 0,
        "learning_rate": 0,
        "momentum": 0,
        "number_of_epochs": 0,
        "number_of_hidden_layers": 0,
        "number_of_neurons_in_layers": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        "dropout_layers": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        "prune_amount": 0,
        "prune_epoch_interval": 0,
        "weight_decay": 0,
        "weight_perturbation_amount": 0,
        "weight_perturbation_interval": 0
    }
    print("Please enter a path to the folder which has the setting files:")
    folder_path = input().strip()

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json") and not file_name.startswith("Generic"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        data['__file_name__'] = file_name
                        settings.append(data)
                except Exception as e:
                    print(f"Failed to load {file_name}: {e}")
        for setting in settings:
            for key in generic_nn_settings:
                if isinstance(setting[key], list):
                    for counter in range(len(setting[key])):
                        generic_nn_settings[key][counter] += setting[key][counter]
                elif isinstance(setting[key], (int, float)):
                    generic_nn_settings[key] += setting[key]
        for key in generic_nn_settings:
            if isinstance(generic_nn_settings[key], list):
                for counter in range(len(generic_nn_settings[key])):
                    if isinstance(generic_nn_settings[key][counter], int):
                        generic_nn_settings[key][counter] = round(generic_nn_settings[key][counter] / len(settings))
                    else:
                        generic_nn_settings[key][counter] = round(generic_nn_settings[key][counter] / len(settings), 3)
            elif isinstance(generic_nn_settings[key], int):
                generic_nn_settings[key] = round(generic_nn_settings[key]/len(settings))
            else:
                generic_nn_settings[key] = round(generic_nn_settings[key] / len(settings),3)
        print(generic_nn_settings)
        save_nn_settings(generic_nn_settings, "Generic", "")
    else:
        print("Invalid folder path provided.")