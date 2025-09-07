import json
import os

from Utils.fileHandler import saveSettings


def createGenericNNSetting():
    settings = []
    genericNNSettings ={
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
    folderPath = input().strip()

    if os.path.exists(folderPath) and os.path.isdir(folderPath):
        for file_name in os.listdir(folderPath):
            if file_name.endswith(".json") and not file_name.startswith("Generic"):
                file_path = os.path.join(folderPath, file_name)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        data['__file_name__'] = file_name
                        settings.append(data)
                except Exception as e:
                    print(f"Failed to load {file_name}: {e}")
        for setting in settings:
            for key in genericNNSettings:
                if isinstance(setting[key], list):
                    for counter in range(len(setting[key])):
                        genericNNSettings[key][counter] += setting[key][counter]
                elif isinstance(setting[key], (int, float)):
                    genericNNSettings[key] += setting[key]
        for key in genericNNSettings:
            if isinstance(genericNNSettings[key], list):
                for counter in range(len(genericNNSettings[key])):
                    if isinstance(genericNNSettings[key][counter], int):
                        genericNNSettings[key][counter] = round(genericNNSettings[key][counter] / len(settings))
                    else:
                        genericNNSettings[key][counter] = round(genericNNSettings[key][counter] / len(settings), 3)
            elif isinstance(genericNNSettings[key], int):
                genericNNSettings[key] = round(genericNNSettings[key]/len(settings))
            else:
                genericNNSettings[key] = round(genericNNSettings[key] / len(settings),3)
        print(genericNNSettings)
        saveSettings(genericNNSettings, "Generic", "")
    else:
        print("Invalid folder path provided.")