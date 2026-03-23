import random
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ModelTrainer.nnTrainer import train_nn
from Utils.constants import *
from Utils.datasetHandler import create_subsets, load_dataset, create_subsets_with_seeds, load_subset
from Utils.fileHandler import load_meta_features_dataset, save_data_frame, get_latest_settings
from Utils.timeFormatter import format_duration


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
            subsets, meta_features, return_seeds, subset_category_columns, subset_file_paths = create_subsets_with_seeds(seed["name"],
                                                                                                                         number_of_instances,
                                                                                                                         seed["classSeeds"],
                                                                                                                         seed["featuresSeeds"],
                                                                                                                         seed["instancesSeeds"],
                                                                                                                         seed["datasetSettings"])
        else:
            subsets, meta_features, return_seeds, subset_category_columns, subset_file_paths = create_subsets(seed["name"],
                                                                                                              number_of_instances,
                                                                                                              seed["datasetSettings"],
                                                                                                              False)
        for subset, meta_feature, return_seed, category_columns, subset_file_path in zip(subsets,
                                                                                         meta_features,
                                                                                         return_seeds,
                                                                                         subset_category_columns,
                                                                                         subset_file_paths):
            meta_feature_dataset.append({
                "dataset_name": seed["name"],
                "seed": return_seed["seed"],
                "subset_type": return_seed["subsetType"],
                "file_name": subset_file_path,
                **meta_feature
            })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(meta_feature_dataset).to_csv(f"{OUTPUT_PATH}SubsetMetaFeatures_{timestamp}.csv", index=False)

def recreate_dataset(subset_dataset, dataset_names, indexes, output_path, number_of_folds, datasets_settings):
    dataset, output_path = load_meta_features_dataset(output_path)
    seeds = []
    for name, group in subset_dataset.groupby('dataset_name'):
        if name in dataset_names:
            settings = get_latest_settings(name)
            dataset_settings =  next((item for item in datasets_settings if item["name"] == name), None)
            seed = {
                "name": name,
                "dataset_settings": dataset_settings,
                "nn_settings": settings,
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
            training_set, testing_set, subset_category_columns = load_subset(row["file_path"], row["seed"], seed["dataset_settings"])
            meta_feature = meta_features.iloc[row["index"]]
            instance, duration = create_instance(seed["name"], seed["nn_settings"], number_of_folds, training_set, testing_set, meta_feature, row, subset_category_columns, row["file_path"])
                # create_instance(seed["name"],
                #                                  settings,
                #                                  number_of_folds,
                #                                  training_set,
                #                                  testing_set,
                #                                  meta_feature,
                #                                  row,
                #                                  subset_category_columns)
            total_duration += duration
            dataset = pd.concat([dataset, instance], ignore_index=True)
            save_data_frame(dataset, output_path)
            predicted_duration = total_duration/counter * (len(dataset_names))
            print(f"{counter} instance created from the {seed["name"]} dataset subset. "
                  f"It took {format_duration(total_duration)}/{format_duration(predicted_duration)}")
            counter+=1

def create_dataset(database_name, output_path, number_of_instances, number_of_folds, dataset_settings):
    dataset, output_path = load_meta_features_dataset(output_path)
    settings = get_latest_settings(database_name)
    total_duration = 0

    if number_of_instances > 1:
        training_sets, testing_sets, meta_features, seeds, subset_category_columns, subset_file_paths = create_subsets(database_name,
                                                                                                    number_of_instances,
                                                                                                    dataset_settings)
    else:
        training_sets, testing_sets, meta_features, seeds, subset_category_columns, subset_file_paths = load_dataset(dataset_settings)

    counter = 0

    for training_set, testing_set, meta_feature, seed, category_columns, subset_file_path in zip(training_sets,
                                                                                                 testing_sets,
                                                                                                 meta_features,
                                                                                                 seeds,
                                                                                                 subset_category_columns,
                                                                                                 subset_file_paths):
        instance, duration = create_instance(database_name,
                                             settings,
                                             number_of_folds,
                                             training_set,
                                             testing_set,
                                             meta_feature,
                                             seed,
                                             category_columns,
                                             subset_file_path)
        total_duration += duration
        dataset = pd.concat([dataset, instance], ignore_index=True)
        save_data_frame(dataset, output_path)
        counter+=1
        predicted_duration = total_duration / counter * number_of_instances
        print(f"{counter} instance created. It took {format_duration(total_duration)}/{format_duration(predicted_duration)}")
    return output_path

def create_instance(dataset_name, settings, number_of_folds, training_set, testing_set, meta_feature, seed, category_columns, subset_file_path):
    start_time = time.time()
    print("")
    print("Dataset name: " + dataset_name)
    print("Seed: " + str(seed["seed"]))
    # Add dataset name, seed and meta feature
    instance_json_object= {
        "dataset_name": dataset_name,
        "seed": seed["seed"],
        "subset_type": seed["subsetType"],
        "file_name": subset_file_path
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
    for config in REGULARISATION_TECHNIQUES:
        print(config["param"])
        training_loss_values, training_accuracies, testing_loss_values, testing_accuracies = train_nn(settings,
                                                                                                      config["param"],
                                                                                                      training_set,
                                                                                                      testing_set,
                                                                                                      seed,
                                                                                                      category_columns,
                                                                                                      number_of_folds)

        instance_json_object[config['fileName']+"_training_loss"] = training_loss_values
        instance_json_object[config['fileName']+"_training_accuracies"] = training_accuracies
        instance_json_object[config['fileName']+"_testing_loss"] = testing_loss_values
        instance_json_object[config['fileName']+"_testing_accuracies"] = testing_accuracies

        if best_training_loss > np.mean(training_loss_values):
            best_training_loss = np.mean(training_loss_values)
            best_training_technique = config['fileName']

        if best_testing_loss > np.mean(testing_loss_values):
            best_testing_loss = np.mean(testing_loss_values)
            best_testing_technique = config['fileName']

    instance_json_object["best_training_technique"] = best_training_technique
    print("best training technique: "+best_training_technique)

    instance_json_object["best_testing_technique"] = best_testing_technique
    print("best testing technique: "+best_testing_technique)
    endTime = time.time()
    duration = endTime - start_time

    # Convert to DataFrame
    return pd.DataFrame([instance_json_object]), duration