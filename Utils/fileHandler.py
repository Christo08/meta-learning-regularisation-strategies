import json
import os

import numpy as np
import pandas as pd

from datetime import datetime

from Models.NN.Errors.fileNotFound import FileNotFound

currentSettingsFilePath = ""

def loadMetaFeaturesDataset(path):
    if os.path.exists(path) and os.path.isfile(path):
        return pd.read_csv(path), path
    elif os.path.exists(path) and os.path.isdir(path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fileName = f"regularisation_{timestamp}.csv"
        filePath = path + "\\" + fileName
        return pd.DataFrame(), filePath
    else:
        raise FileNotFound(f"This is not a valid path {path}")

def loadDatasetSetting():
    with open('Data/Datasets/Input/dataset_info.json', 'r') as file:
        data = json.load(file)
    return data

def saveMetaFeaturesDataset(dataset, filePath):
    dataset.to_csv(filePath, index=False)

def loadMetaFeaturesCSV():
    path = input("Enter the path of the meta features dataset file:")
    if os.path.exists(path) and os.path.isfile(path):
        return pd.read_csv(path, sep=",", quotechar='"')
    else:
        raise FileNotFound(f"This is not a valid path {path}")

def saveRunsDataset(runsPath, runsDataset):
    runsDataset.to_csv(runsPath, index=False)

def loadRunsDataset(path):
    if os.path.exists(path) and os.path.isfile(path):
        return pd.read_csv(path), ""
    elif os.path.exists(path) and os.path.isdir(path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fileName = f"runs_{timestamp}.csv"
        return pd.DataFrame(), fileName
    else:
        raise FileNotFound(f"This is not a valid path {path}")

def saveSettings(settings, datasetName, path):
    if path == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fileName = f"{datasetName}_nn_setting_{timestamp}.json"
        newPath = f"Data/Settings/NNSettings/{fileName}"
        with open(newPath, "x") as file:
            json.dump(settings, file, indent=4, cls=ObjectEncoder)
        return newPath
    else:
        oldSettings = loadSettings(path)
        newSettings = {**oldSettings, **settings}
        with open(path, "w") as file:
            json.dump(newSettings, file, indent=4, cls=ObjectEncoder)
        return path

def loadSettings(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        raise

def saveModuleSettings(settings, moduleType, technique):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName = f"{technique}_setting_{timestamp}.json"
    currentSettingsFilePath = f"Data/Settings/ModelSettings/{moduleType}/{fileName}"
    with open(currentSettingsFilePath, "x") as file:
        json.dump(settings, file, indent=4, cls=ObjectEncoder)

def loadModuleSettings(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        raise

class ObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray) or isinstance(obj, dict) or isinstance(obj, list):  # If it's a dictionary, apply recursively
            return  self.convert_ndarray_to_list(obj)
        else:
            return super().default(obj)

    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, np.ndarray):  # Check if it's a numpy array
            return obj.tolist()  # Convert ndarray to list
        elif isinstance(obj, dict):  # If it's a dictionary, apply recursively
            return {key: self.convert_ndarray_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):  # If it's a list, apply recursively
            return [ self.convert_ndarray_to_list(item) for item in obj]
        else:
            return obj