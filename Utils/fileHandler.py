import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime
from Utils.Errors.fileNotFound import FileNotFound
from Utils.constants import *

current_settings_file_path = ""

def load_meta_features_dataset(path):
    if os.path.exists(path) and os.path.isfile(path):
        return pd.read_csv(path), path
    elif os.path.exists(path) and os.path.isdir(path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"regularisation_{timestamp}.csv"
        file_path = path + "\\" + file_name
        return pd.DataFrame(), file_path
    else:
        raise FileNotFound(f"This is not a valid path {path}")

def load_dataset_setting_file(path):
    with open(path, 'r') as file:
        data_settings = json.load(file)
    return data_settings

def save_data_frame(data_frame, file_path):
    data_frame.to_csv(file_path, index=False)

def load_meta_features_csv(type):
    if type == "":
        path = input(f"Enter the path of the meta features dataset file:")
    else:
        path = input(f"Enter the path of the {type} meta features dataset file or folder:")
    if os.path.exists(path) and os.path.isfile(path):
        return pd.read_csv(path, sep=",", quotechar='"')
    else:
        raise FileNotFound(f"This is not a valid path {path}")

def load_results_csv():
    file_path = input("What is the path to the result file?")
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return pd.read_csv(file_path, sep=",", quotechar='"')
    else:
        raise FileNotFound(f"This is not a valid path {file_path}")


def save_nn_settings(settings, dataset_name, path):
    if path == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{dataset_name.replace(" ","_")}_nn_setting_{timestamp}.json"
        new_path = f"{BASIC_NN_SETTINGS_PATH}/{file_name}"
        with open(new_path, "x") as file:
            json.dump(settings, file, indent=4, cls=ObjectEncoder)
        return new_path
    else:
        old_settings = load_settings(path)
        new_settings = {**old_settings, **settings}
        with open(path, "w") as file:
            json.dump(new_settings, file, indent=4, cls=ObjectEncoder)
        return path

def save_meta_learner_settings(settings, module_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if module_type == "DecisionTrees":
        file_name = f"Decision_trees_setting_{timestamp}.json"
    elif module_type == "RandomForest":
        file_name = f"Random_forest_setting_{timestamp}.json"
    elif module_type == "KNearestNeighbors":
        file_name = f"knn_setting_{timestamp}.json"
    elif module_type == "SupportVectorMachines":
        file_name = f"svm_setting_{timestamp}.json"
    else:
        file_name = f"nn_setting_{timestamp}.json"
    path = f"{META_LEARNER_SETTINGS_PATH}/{module_type}/{file_name}"
    with open(path, "x") as file:
        json.dump(settings, file, indent=4, cls=ObjectEncoder)

def load_settings(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        raise

def get_latest_settings(name):
    directory = Path(BASIC_NN_SETTINGS_PATH)
    normalized_prefix = name.strip().replace(" ", "_")

    matching_files = [
        file
        for file in directory.glob(f"{normalized_prefix}_nn_setting_*.json")
    ]

    if not matching_files:
        return None

    file_name = max(
        matching_files,
        key=lambda f: extract_timestamp(f.name),
    )

    return load_settings(BASIC_NN_SETTINGS_PATH+"/"+file_name.name)

def extract_timestamp(filename):
    try:
        return filename.split("_nn_setting_")[1].replace(".json", "")
    except IndexError as exc:
        raise ValueError(f"Invalid filename format: {filename}") from exc

def save_subset(subset, seed, dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{seed}_{timestamp}.csv"
    folder_path = f"{SUBSET_PATH}/{dataset_name}"
    file_path = f"{folder_path}/{file_name}"

    os.makedirs(folder_path, exist_ok=True)
    try:
        subset.to_csv(file_path, index=False)
    except (OSError, IOError, PermissionError) as e:
        print(f"Error saving subset to CSV at '{file_path}': {e}")
        raise
    return file_path

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