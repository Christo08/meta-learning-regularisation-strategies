import random
from datetime import datetime

import joblib
import pandas as pd
import torch

from src.Models.NN.network import Network
from src.ModelTrainer.decisionTreeTrainer import training_meta_decision_trees
from src.ModelTrainer.knnTrainer import training_meta_k_nearest_neighbors
from src.ModelTrainer.nnTrainer import training_meta_nns
from src.ModelTrainer.randomForestTrainer import training_meta_random_forests
from src.ModelTrainer.svmTrainer import training_meta_support_vector_machines
from src.Utils.constants import META_LEARN_TYPES
from src.Utils.datasetHandler import load_full_dataset
from src.Utils.fileHandler import save_data_frame, folder_maker, load_json_file, get_latest_settings, load_results_csv
from src.Utils.menus import show_meta_leaner_type_menu
from src.Utils.metaFeatureCalculator import calculate_meta_features


def train_meta_learners(training_dataset, testing_dataset):
    training_dataset.drop(columns=["dataset_name"], inplace=True)
    testing_dataset.drop(columns=["dataset_name"], inplace=True)
    selected_meta_learn_types = show_meta_leaner_type_menu()
    number_of_folds = int(input("How many folds do you want the meta-learner to getrained? "))
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

def test_meta_learner(name, dataset_settings):
    return
#     number_of_folds = int(input("How many folds do you want to use per dataset? "))
#
#     settings_file_path = input(f"Enter the path of the settings index file: ")
#     meta_learners_results = load_results_csv()
#     output_path = input("Enter the path of the Output stats folder: ")
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     seed = random.randint(0, 4294967295)
#     sets, category_columns = load_full_dataset(seed, dataset_settings)
#     training_set = sets[0]
#     testing_set = sets[1]
#
#     nn_settings = get_latest_settings(name)
#
#     meta_learners_performs =predicted_best_techniques(seed, meta_learners_results, dataset, category_columns)
#
#
# def predicted_best_techniques(seed, meta_learners_results, dataset, category_columns):
#     meta_features = calculate_meta_features(dataset, category_columns)
#
#     techniques = list(meta_learners_results["technique"].dropna().unique())
#     model_types = list(meta_learners_results["model type"].dropna().unique())
#
#     techniques_predicted = {technique: [] for technique in techniques}
#
#     for technique in techniques:
#         meta_learners_results_per_technique= meta_learners_results[meta_learners_results["technique"] == technique]
#         for model in model_types:
#             meta_learners_results_per_technique_and_model = meta_learners_results_per_technique[meta_learners_results_per_technique["model type"] == model]
#             if model == "nn":
#                 checkpoint = torch.load(meta_learners_results_per_technique_and_model["model path"])
#                 model = Network(**checkpoint["model_kwargs"])
#                 model.load_state_dict(checkpoint["state_dict"])
#                 model.eval()
#
#                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#                 model = model.to(device)
#
#                 input_data = torch.tensor(meta_features, dtype=torch.float32).to(device)
#                 with torch.no_grad():
#                     is_best = model(input_data)
#
#             else:
#                 module = joblib.load(meta_learners_results_per_technique_and_model["model path"])
#                 is_best = module.predict(meta_features)
#             if is_best:
#                 techniques_predicted[technique].append(meta_learners_results_per_technique_and_model["testing f1"])
#
#     best_technique = []
#     best_count = 0
#     for technique, preforms in techniques_predicted.items():
#         count = len(preforms)
#
#         if count > best_count:
#             best_technique = [technique]
#         elif count == best_count:
#             best_technique.append(technique)
#
#     if len(best_technique) == 1:
#         return best_technique[0]
#     else:
#         tied_techniques = best_technique.copy()
#         best_technique = []
#         best_preforms = 0
#         for technique in tied_techniques:
#             preforms_sum = sum(techniques_predicted[technique])
#             if preforms_sum > best_preforms:
#                 best_preforms = [preforms_sum]
#             elif preforms_sum == best_preforms:
#                 best_technique.append(technique)
#         if len(best_technique) == 1:
#             return best_technique[0]
#         elif "baseline" in best_technique:
#             return "baseline"
#         else:
#             random.seed(seed)
#             index = random.randint(0, len(best_technique) - 1)
#             return best_technique[index]