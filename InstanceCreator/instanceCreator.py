import os
import random
import time

import numpy as np
import pandas as pd

from datetime import datetime

from ModelTrainer.nnTrainer import train_nn
from Utils.datasetHandler import create_subsets, load_dataset, create_subsets_with_seeds, load_subset
from Utils.fileHandler import load_meta_features_dataset, save_data_frame, load_settings
from Utils.timeFormatter import format_duration

configurations = [
    {"name": "baseline", "param": "baseline", "fileName": "baseline"},
    {"name": "batchNormalisation", "param": "batchNormalisation", "fileName": "batch_normalisation"},
    {"name": "dropout", "param": "dropout", "fileName": "dropout"},
    {"name": "layerNormalisation", "param": "layerNormalisation", "fileName": "layer_normalisation"},
    {"name": "SMOTE", "param": "SMOTE", "fileName": "SMOTE"},
    {"name": "prune", "param": "prune", "fileName": "prune"},
    {"name": "weightDecay", "param": "weightDecay", "fileName": "weight_decay"},
    {"name": "weightNormalisation", "param": "weightNormalisation", "fileName": "weight_normalisation"},
    {"name": "weightPerturbation", "param": "weightPerturbation", "fileName": "weight_perturbation"}
]

def recreate_subsets(meta_feature_dataset, number_of_instances, datasets_settings, names=None):
    if names is None:
        names = []
    seeds = []
    if len(meta_feature_dataset)>0:
        for name, group in meta_feature_dataset.groupby('dataset_name'):
            seed ={
                "name": name,
                "datasetSettings": next((item for item in datasets_settings if item["name"] == name), None),
                "classSeeds": [],
                "featuresSeeds": [],
                "instancesSeeds": [],
                "isComplete": True
            }
            if group.shape[0] < number_of_instances:
                seed["isComplete"] = False
            else:
                seed["isComplete"] = True
                for index, row in group.iterrows():
                    if row['subset_type'] == "classes":
                        seed["classSeeds"].append({
                            "seed": row['seed'],
                            "subsetType": row['subset_type']
                        })
                    elif row['subset_type'] == "instances":
                        seed["instancesSeeds"].append({
                            "seed": row['seed'],
                            "subsetType": row['subset_type']
                        })
                    else:
                        seed["featuresSeeds"].append({
                            "seed": row['seed'],
                            "subsetType": row['subset_type']
                        })
            seeds.append(seed)
    else:
        for name in names:
            seed ={
                "name": name,
                "datasetSettings": next((item for item in datasets_settings if item["name"] == name), None),
                "isComplete": False
            }
            seeds.append(seed)

    meta_feature_dataset = []
    for seed in seeds:
        if seed["isComplete"]:
            subsets, meta_features, return_seeds, subset_category_columns = create_subsets_with_seeds(seed["name"],
                                                                                                      number_of_instances,
                                                                                                      seed["classSeeds"],
                                                                                                      seed["featuresSeeds"],
                                                                                                      seed["instancesSeeds"],
                                                                                                      seed["datasetSettings"])
        else:
            subsets, meta_features, return_seeds, subset_category_columns = create_subsets(seed["name"],
                                                                                           number_of_instances,
                                                                                           seed["datasetSettings"],
                                                                                           False)
        for subset, meta_feature, return_seed, category_columns in zip(subsets,
                                                                       meta_features,
                                                                       return_seeds,
                                                                       subset_category_columns):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{return_seed['seed']}_{timestamp}.csv"
            folder_path = f"Data/Datasets/Input/Subsets/{seed['name']}"
            file_path = f"{folder_path}/{file_name}"

            os.makedirs(folder_path, exist_ok=True)
            subset.to_csv(file_path, index=False)

            meta_feature_dataset.append({
                "dataset_name": seed["name"],
                "seed": return_seed["seed"],
                "subset_type": return_seed["subsetType"],
                "file_name": file_path,
                **meta_feature
            })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(meta_feature_dataset).to_csv(f"Data/Datasets/Output/Raw/SubsetMetaFeatures_{timestamp}.csv", index=False)

def recreate_dataset(subset_dataset, dataset_names, indexes, settings_file_path, output_path, number_of_folds):
    dataset, output_path = load_meta_features_dataset(output_path)
    settings = load_settings(settings_file_path)
    seeds = []
    for name, group in subset_dataset.groupby('dataset_name'):
        if name in dataset_names:
            seed = {
                "name": name,
                "rows": [],
            }
            for index, row in group.iterrows():
                seed["rows"].append({
                    "seed": row['seed'],
                    "subsetType": row['subset_type'],
                    "file_path": row["file_name"],
                    "index": index
                })
            seeds.append(seed)
    total_duration = 0
    meta_features = subset_dataset.drop(columns=["dataset_name", "seed", "subset_type", "file_name"])
    for seed in seeds:
        counter = 1
        for index in indexes:
            row = seed["rows"][index]
            training_set, testing_set, subset_category_columns = load_subset(row["file_path"], seed["name"], row["seed"])
            meta_feature = meta_features.iloc[row["index"]]
            instance, duration = create_instance(seed["name"],
                                                 settings,
                                                 number_of_folds,
                                                 training_set,
                                                 testing_set,
                                                 meta_feature,
                                                 row,
                                                 subset_category_columns)
            total_duration += duration
            dataset = pd.concat([dataset, instance], ignore_index=True)
            save_data_frame(dataset, output_path)
            predicted_duration = total_duration/counter * (len(dataset_names))
            print(f"{counter} instance created from the {seed["name"]} dataset subset. "
                  f"It took {format_duration(total_duration)}/{format_duration(predicted_duration)}")
            counter+=1

def create_dataset(database_name, output_path, number_of_instances, settings_file_path, number_of_folds, dataset_settings):
    dataset, output_path = load_meta_features_dataset(output_path)
    settings = load_settings(settings_file_path)
    total_duration = 0

    if number_of_instances > 1:
        training_sets, testing_sets, meta_features, seeds, subset_category_columns = create_subsets(database_name,
                                                                                                    number_of_instances,
                                                                                                    dataset_settings)
    else:
        training_sets, testing_sets, meta_features, seeds, subset_category_columns = load_dataset(dataset_settings)

    counter = 0

    for training_set, testing_set, meta_feature, seed, category_columns in zip(training_sets,
                                                                               testing_sets,
                                                                               meta_features,
                                                                               seeds,
                                                                               subset_category_columns):
        instance, duration = create_instance(database_name,
                                             settings,
                                             number_of_folds,
                                             training_set,
                                             testing_set,
                                             meta_feature,
                                             seed,
                                             category_columns)
        total_duration += duration
        dataset = pd.concat([dataset, instance], ignore_index=True)
        save_data_frame(dataset, output_path)
        counter+=1
        predicted_duration = total_duration / counter * number_of_instances
        print(f"{counter} instance created. It took {format_duration(total_duration)}/{format_duration(predicted_duration)}")
    return output_path

def create_instance(dataset_name, settings, number_of_folds, training_set, testing_set, meta_feature, seed, category_columns):
    start_time = time.time()
    print("")
    print("Dataset name: " + dataset_name)
    print("Seed: " + str(seed["seed"]))
    # Add dataset name, seed and meta feature
    instance_json_object= {
        "dataset_name": dataset_name,
        "seed": seed["seed"],
        "subset_type": seed["subsetType"]
    }
    instance_json_object = {**instance_json_object, **meta_feature}
    best_training_loss = float('inf')
    best_training_technique = ""
    best_testing_loss = float('inf')
    best_testing_technique = ""
    random.seed(seed["seed"])
    seed = random.randint(0, 2**32 - 1)
    random.seed(seed)

    # Perform training for each configuration
    for config in configurations:
        print(config["param"])
        training_losses, testing_losses = train_nn(settings,
                                                   config["param"],
                                                   training_set,
                                                   testing_set,
                                                   seed,
                                                   category_columns,
                                                   number_of_folds)

        instance_json_object[config['fileName']+"_training_loss"] = training_losses
        instance_json_object[config['fileName']+"_testing_loss"] = testing_losses

        if best_training_loss > np.mean(training_losses):
            best_training_loss = np.mean(training_losses)
            best_training_technique = config['fileName']

        if best_testing_loss > np.mean(testing_losses):
            best_testing_loss = np.mean(testing_losses)
            best_testing_technique = config['fileName']

    instance_json_object["best_training_technique"] = best_training_technique
    print("best training technique: "+best_training_technique)

    instance_json_object["best_testing_technique"] = best_testing_technique
    print("best testing technique: "+best_testing_technique)
    endTime = time.time()
    duration = endTime - start_time

    # Convert to DataFrame
    return pd.DataFrame([instance_json_object]), duration