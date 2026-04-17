import random
from datetime import datetime

import joblib
import pandas as pd
import torch

from src.ModelTrainer.decisionTreeTrainer import training_meta_decision_trees
from src.ModelTrainer.knnTrainer import training_meta_k_nearest_neighbors
from src.ModelTrainer.nnTrainer import training_meta_nns, train_basic_nns
from src.ModelTrainer.randomForestTrainer import training_meta_random_forests
from src.ModelTrainer.svmTrainer import training_meta_support_vector_machines
from src.Models.NN.network import Network
from src.Utils.constants import *
from src.Utils.datasetHandler import load_full_dataset, splitSet
from src.Utils.fileHandler import save_data_frame, folder_maker, load_json_file, get_latest_settings
from src.Utils.menus import show_meta_leaner_type_menu
from src.Utils.metaFeatureCalculator import calculate_meta_features
from src.Utils.metaFeatureDatasetHandler import prepare_meta_feature_full_dataset_for_states, add_hyperparameters


def train_meta_learners(training_dataset, testing_dataset):
    training_dataset.drop(columns=["dataset_name"], inplace=True)
    testing_dataset.drop(columns=["dataset_name"], inplace=True)
    selected_meta_learn_types = show_meta_leaner_type_menu()
    number_of_folds = int(input("How many folds do you want the meta-learner to get trained? "))
    results = pd.DataFrame(columns=["model type", "technique",  "training loses", "testing loses"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings_file_path = input(f"Enter the path of the settings index file: ")
    output_path = input("Enter the path of the output stats folder: ")
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

def test_meta_learner(dataset_name, dataset_settings, meta_learners_results, number_of_folds, transformer_path):
    seed = random.randint(0, 4294967295)
    random.seed(seed)
    dataset, category_columns = load_full_dataset(seed, dataset_settings, False)
    sets = splitSet(dataset, seed)
    training_set = sets[0]
    testing_set = sets[1]

    nn_settings = get_latest_settings(dataset_name)

    best_technique = predict_best_technique(meta_learners_results,
                                            dataset,
                                            category_columns,
                                            transformer_path,
                                            nn_settings)
    instance_json_object = train_nns(dataset_name,
                                     best_technique,
                                     seed,
                                     training_set,
                                     testing_set,
                                     category_columns,
                                     number_of_folds,
                                     nn_settings,
                                     meta_learners_results,
                                     transformer_path)

    return pd.DataFrame([instance_json_object])

def predict_best_technique(meta_learners_results, dataset, category_columns, transformer_path, nn_settings):
    meta_features = pd.DataFrame([calculate_meta_features(dataset, category_columns)])
    meta_features = add_hyperparameters(meta_features, nn_settings)
    meta_features = prepare_meta_feature_full_dataset_for_states(meta_features, transformer_path)

    techniques = list(meta_learners_results["technique"].dropna().unique())
    model_types = list(meta_learners_results["model type"].dropna().unique())

    techniques_predicted = {technique.replace(" ", "_") : [] for technique in techniques}
    for technique in techniques:
        meta_learners_results_per_technique = meta_learners_results[
            meta_learners_results["technique"].replace(" ", "_") == technique]
        best_metrix = -1
        best_model_types = []

        for model_type in model_types:
            meta_learners_results_per_technique_and_model = meta_learners_results_per_technique[
                meta_learners_results_per_technique["model type"] == model_type]
            if not meta_learners_results_per_technique_and_model.empty:
                trues = (meta_learners_results_per_technique_and_model["training true positives"].iloc[0] +
                         meta_learners_results_per_technique_and_model["training true negatives"].iloc[0])
                total = trues + (meta_learners_results_per_technique_and_model["training false positives"].iloc[0]+
                                meta_learners_results_per_technique_and_model["training false negatives"].iloc[0])
                metrix = trues/total*100
                if metrix > best_metrix:
                    best_metrix = metrix
                    best_model_types = [model_type]
                elif metrix == best_metrix:
                    best_model_types.append(model_type)
        best_model_type = best_model_types[random.randint(0, len(best_model_types) - 1)]
        best_model_row = meta_learners_results_per_technique[
            meta_learners_results_per_technique["model type"] == best_model_type]
        path = best_model_row["model path"].values[0]
        if best_model_type == "Neural Network":
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
                        techniques_predicted[technique.replace(" ", "_")].append(best_model_row["testing f1"].values[0])
        else:
            model = joblib.load(path)
            is_best = model.predict(meta_features)
            if best_model_type == "svm":
                if is_best[0] == 1:
                    techniques_predicted[technique.replace(" ", "_")].append(best_model_row["testing f1"].values[0])
            else:
                if is_best[0][1]:
                    techniques_predicted[technique.replace(" ", "_")].append(best_model_row["testing f1"].values[0])
    best_technique = []
    best_count = 0
    best_f1_score = 0
    for technique in techniques_predicted:
        count = len(techniques_predicted[technique])
        f1_score = 0
        if count > 0:
            f1_score = max(techniques_predicted[technique])
        if count > best_count:
            best_count = count
            best_technique = [technique]
            best_f1_score = f1_score
        elif count == best_count and best_f1_score < f1_score:
            best_technique = [technique]
            best_f1_score = f1_score
        elif count == best_count and best_f1_score == f1_score:
            best_technique.append(technique)
    if len(best_technique) == 1:
        return best_technique[0]
    elif len(best_technique) == 0 or "baseline" in best_technique:
        return "baseline"
    else:
        return best_technique[random.randint(0, len(best_technique) - 1)]

def train_nns(dataset_name, best_technique, seed, training_set, testing_set, category_columns, number_of_folds, nn_settings, meta_learners_results, path_to_transformer):
    print("")
    print("Dataset name: " + dataset_name)
    print("Seed: " + str(seed))
    print("Predict best technique: " +best_technique)
    # Add dataset name, seed and meta feature
    instance_json_object = {
        "dataset_name": dataset_name,
        "seed": seed,
        "best_technique": best_technique
    }
    for config in REGULARISATION_TECHNIQUES:
        print(config["param"])
        matrices = train_basic_nns(nn_settings, config["param"], training_set, testing_set, seed, category_columns,
                                   number_of_folds)

        instance_json_object[f"{config['fileName']}_training_loss"] = matrices["training_loss"]
        instance_json_object[f"{config['fileName']}_training_accuracies"] = matrices["training_accuracies"]
        instance_json_object[f"{config['fileName']}_training_f1_scores"] = matrices["training_f1_scores"]
        instance_json_object[f"{config['fileName']}_testing_loss"] = matrices["testing_loss"]
        instance_json_object[f"{config['fileName']}_testing_accuracies"] = matrices["testing_accuracies"]
        instance_json_object[f"{config['fileName']}_testing_f1_scores"] = matrices["testing_f1_scores"]

    instance_json_object["meta_learner_training_loss"] =  instance_json_object[f"{best_technique}_training_loss"]
    instance_json_object["meta_learner_training_accuracies"] = instance_json_object[f"{best_technique}_training_accuracies"]
    instance_json_object["meta_learner_training_f1_scores"] = instance_json_object[f"{best_technique}_training_f1_scores"]
    instance_json_object["meta_learner_testing_loss"] = instance_json_object[f"{best_technique}_testing_loss"]
    instance_json_object["meta_learner_testing_accuracies"] = instance_json_object[f"{best_technique}_testing_accuracies"]
    instance_json_object["meta_learner_testing_f1_scores"] = instance_json_object[f"{best_technique}_testing_f1_scores"]

    return instance_json_object