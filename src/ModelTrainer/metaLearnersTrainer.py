import joblib
import pandas as pd
import random
import torch
from datetime import datetime

from src.ModelTrainer.nnTrainer import train_basic_nns
from src.ModelTrainer.decisionTreeTrainer import training_meta_decision_trees
from src.ModelTrainer.knnTrainer import training_meta_k_nearest_neighbors
from src.ModelTrainer.nnTrainer import training_meta_nns
from src.ModelTrainer.randomForestTrainer import training_meta_random_forests
from src.ModelTrainer.svmTrainer import training_meta_support_vector_machines
from src.Models.NN.network import Network
from src.Utils.datasetHandler import load_full_dataset, splitSet
from src.Utils.fileHandler import save_data_frame, folder_maker, load_json_file, get_latest_settings
from src.Utils.menus import show_meta_leaner_type_menu
from src.Utils.metaFeatureCalculator import calculate_meta_features
from src.Utils.metaFeatureDatasetHandler import prepare_meta_feature_full_dataset_for_states
from src.Utils.constants import *


def train_meta_learners(training_dataset, testing_dataset):
    training_dataset.drop(columns=["dataset_name"], inplace=True)
    testing_dataset.drop(columns=["dataset_name"], inplace=True)
    selected_meta_learn_types = show_meta_leaner_type_menu()
    number_of_folds = int(input("How many folds do you want the meta-learner to get trained? "))
    results = pd.DataFrame(columns=["model type", "technique",  "training loses", "testing loses"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings_file_path = input(f"Enter the path of the settings index file: ")
    output_path = input("Enter the path of the Output stats folder: ")
    settings = load_json_file(settings_file_path)

    for selected_meta_learn_type in selected_meta_learn_types:
        seed = random.randint(0, 4294967295)
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            settings_file_path = settings["DecisionTrees"]
            result = training_meta_decision_trees(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            settings_file_path = settings["RandomForest"]
            result = training_meta_random_forests(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            settings_file_path = settings["KNearestNeighbors"]
            result = training_meta_k_nearest_neighbors(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            settings_file_path = settings["NeuralNetworks"]
            result = training_meta_nns(settings_file_path, training_dataset, testing_dataset, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            settings_file_path = settings["SupportVectorMachines"]
            result = training_meta_support_vector_machines(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        else:
            return
        results = pd.concat([results, pd.DataFrame(result)], ignore_index=True)
    output_path = f'{output_path}\\{timestamp}'
    folder_maker(output_path)
    file_name = f"{output_path}\\meta_learners_results.csv"
    save_data_frame(results, file_name)

def test_meta_learner(dataset_name, dataset_settings, meta_learners_results, number_of_folds, path_to_transformer):
    seed = random.randint(0, 4294967295)
    dataset, category_columns = load_full_dataset(seed, dataset_settings, False)
    sets =splitSet(dataset, seed)
    training_set = sets[0]
    testing_set = sets[1]

    nn_settings = get_latest_settings(dataset_name)

    meta_learners_performs = predicted_best_techniques(seed, meta_learners_results, dataset, category_columns, path_to_transformer)
    print(f"The best technique is: {meta_learners_performs}")

    instance_json_object = train_nns(dataset_name, meta_learners_performs, seed, training_set, testing_set, category_columns, number_of_folds, nn_settings)

    return pd.DataFrame([instance_json_object])


def predicted_best_techniques(seed, meta_learners_results, dataset, category_columns, path_to_transformer):
    meta_features = pd.DataFrame([calculate_meta_features(dataset, category_columns)])
    meta_features = prepare_meta_feature_full_dataset_for_states(meta_features, path_to_transformer)

    techniques = list(meta_learners_results["technique"].dropna().unique())
    model_types = list(meta_learners_results["model type"].dropna().unique())

    techniques_predicted = {technique.replace(" ", "_") : [] for technique in techniques}

    for technique in techniques:
        meta_learners_results_per_technique= meta_learners_results[meta_learners_results["technique"].replace(" ", "_") == technique]
        for model in model_types:
            meta_learners_results_per_technique_and_model = meta_learners_results_per_technique[meta_learners_results_per_technique["model type"] == model]
            path = meta_learners_results_per_technique_and_model["model path"].values[0]
            if model == "Neural Network":
                checkpoint = torch.load(path)
                model = Network(**checkpoint["model_kwargs"])
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                input_np = meta_features.to_numpy(dtype="float32", copy=False)
                input_data = torch.from_numpy(input_np).to(device)
                with torch.no_grad():
                    is_best = model(input_data)
                    if is_best[0][1] == 1.0:
                        techniques_predicted[technique.replace(" ", "_")].append(meta_learners_results_per_technique_and_model["testing f1"].values[0])

            else:
                module = joblib.load(path)
                is_best = module.predict(meta_features)
                if model == "svm":
                    if is_best[0] == 1:
                        techniques_predicted[technique.replace(" ", "_")].append(meta_learners_results_per_technique_and_model["testing f1"].values[0])
                else:
                    if is_best[0][1]:
                        techniques_predicted[technique.replace(" ", "_")].append(meta_learners_results_per_technique_and_model["testing f1"].values[0])

    best_technique = []
    best_count = 0
    for technique, preforms in techniques_predicted.items():
        count = len(preforms)

        if count > best_count:
            best_technique = [technique]
            best_count = count
        elif count == best_count:
            best_technique.append(technique)

    if len(best_technique) == 1:
        return best_technique[0]
    else:
        tied_techniques = best_technique.copy()
        best_technique = []
        best_preforms = 0
        for technique in tied_techniques:
            preforms_sum = sum(techniques_predicted[technique])/len(techniques_predicted[technique])
            if preforms_sum > best_preforms:
                best_technique = [technique]
                best_preforms = preforms_sum
            elif preforms_sum == best_preforms:
                best_technique.append(technique)
        if len(best_technique) == 1:
            return best_technique[0]
        elif "baseline" in best_technique:
            return "baseline"
        else:
            random.seed(seed)
            index = random.randint(0, len(best_technique) - 1)
            return best_technique[index]

def train_nns(dataset_name, best_technique, seed, training_set, testing_set, category_columns, number_of_folds, nn_settings):
    print("")
    print("Dataset name: " + dataset_name)
    print("Seed: " + str(seed))
    # Add dataset name, seed and meta feature
    instance_json_object = {
        "dataset_name": dataset_name,
        "seed": seed
    }
    for config in REGULARISATION_TECHNIQUES:
        print(config["param"])
        training_loss_values, training_accuracies, testing_loss_values, testing_accuracies = train_basic_nns(nn_settings,
                                                                                                             config["param"],
                                                                                                             training_set,
                                                                                                             testing_set,
                                                                                                             seed,
                                                                                                             category_columns,
                                                                                                             number_of_folds)

        instance_json_object[f"{config['fileName']}_training_loss"] = training_loss_values
        instance_json_object[f"{config['fileName']}_training_accuracies"] = training_accuracies
        instance_json_object[f"{config['fileName']}_testing_loss"] = testing_loss_values
        instance_json_object[f"{config['fileName']}_testing_accuracies"] = testing_accuracies

    instance_json_object["meta_learner_training_loss"] =  instance_json_object[f"{best_technique}_training_loss"]
    instance_json_object["meta_learner_training_accuracies"] = instance_json_object[f"{best_technique}_training_accuracies"]
    instance_json_object["meta_learner_testing_loss"] = instance_json_object[f"{best_technique}_testing_loss"]
    instance_json_object["meta_learner_testing_accuracies"] = instance_json_object[f"{best_technique}_testing_accuracies"]

    return instance_json_object