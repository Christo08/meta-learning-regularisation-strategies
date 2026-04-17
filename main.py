from datetime import datetime

import pandas as pd
import torch

from src.ModelTrainer.metaLearnersTrainer import train_meta_learners, test_meta_learner
from src.Optimisers.metaLearnersOptimiser import optimise_meta_learners
from src.Optimisers.nnOptimiser import optimise_basic_nn
from src.Utils.constants import *
from src.Utils.fileHandler import save_data_frame, load_json_file, load_settings, load_meta_features_csv, \
    load_results_csv
from src.Utils.instanceCreator import create_dataset, recreate_subsets, recreate_dataset
from src.Utils.menus import show_dataset_menu, show_menu
from src.Utils.metaFeatureDatasetHandler import prepare_meta_feature_dataset_for_states, prepare_meta_feature_sets
from src.Utils.statsCalculator import calculate_meta_learners_stats, calculate_dataset_stats


def main():
    print(f"PyTorch version: {torch.__version__}")  # Ensure it's a CUDA-compatible version
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version {torch.version.cuda}")
    else:
        print(f"Device: CPU")
    print(f"")
    datasets_settings = load_json_file(DATASETS_INFO_PATH)
    if not datasets_settings:
        return
    while True:
        process = show_menu("Select process by entering a number: ", PROCESS_OPTIONS)
        if process == PROCESS_OPTIONS[0]:
            while True:
                names = show_dataset_menu(datasets_settings)
                if not names:
                    break
                parameter_group = show_menu("Select parameter group by entering a number:", PARAMETER_GROUPS)
                basic_settings = None
                if not(parameter_group == PARAMETER_GROUPS[0] or parameter_group == PARAMETER_GROUPS[1]):
                    basic_settings = load_settings(input("Enter the path to the basic settings file of the NN:"))
                for name in names:
                    dataset_settings = next((item for item in datasets_settings if item["name"] == name), None)
                    quited = optimise_basic_nn(name, dataset_settings, parameter_group, basic_settings)
                    if quited:
                         break
        elif process == PROCESS_OPTIONS[1]:
            while True:
                names = show_dataset_menu(datasets_settings)
                if not names:
                    break
                output_path = input("Enter the path of the output dataset file or folder: ")
                number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
                number_of_folds = int(input("How many folds do you want to use per instance? "))
                for name in names:
                    dataset_settings = next((item for item in datasets_settings if item["name"] == name), None)
                    output_path = create_dataset(name,
                                                 output_path,
                                                 number_of_instances,
                                                 number_of_folds,
                                                 dataset_settings)
        elif process == PROCESS_OPTIONS[2]:
            if input("Do you have a meta-feature file? (y/n): ").lower() == "y":
                dataset = load_meta_features_csv("")
                names =[]
                number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
                recreate_subsets(dataset, number_of_instances, datasets_settings, names)
            else:
                dataset = pd.DataFrame(columns=["dataset_name","seed","number_of_features","proportion_of_numeric_features",
                                                "number_of_instances","number_of_classes","ratio_of_instances_to_features",
                                                "ratio_of_classes_to_features","ratio_of_instances_to_classes",
                                                "ratio_of_min_to_max_instances_per_class","proportion_of_features_with_outliers",
                                                "average_mutual_information","minimum_mutual_information",
                                                "maximum_mutual_information","equivalent_number_of_features",
                                                "noise_to_signal_ratio_of_features","baseline_training_loss",
                                                "baseline_testing_loss","batch_normalisation_training_loss",
                                                "batch_normalisation_testing_loss","dropout_training_loss","dropout_testing_loss",
                                                "layer_normalisation_training_loss","layer_normalisation_testing_loss",
                                                "prune_training_loss","prune_testing_loss",
                                                "weight_decay_training_loss","weight_decay_testing_loss","weight_normalisation_training_loss",
                                                "weight_normalisation_testing_loss","weight_perturbation_training_loss",
                                                "weight_perturbation_testing_loss","best_training_technique","best_testing_technique",
                                                "subset_type"])
                names = show_dataset_menu(datasets_settings)
                if names:
                    number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
                    recreate_subsets(dataset, number_of_instances, datasets_settings, names)
        elif process == PROCESS_OPTIONS[3]:
            names = show_dataset_menu(datasets_settings)
            if names:
                subset_dataset = load_meta_features_csv("")
                output_path = input("Enter the path of the output dataset file or folder: ")
                number_of_folds = int(input("How many folds do you want to use per instance? "))
                index_to_create = input("Enter the indexes to recreate (separated by commas): ").replace(' ', '').split(",")
                index_to_create = [int(index) for index in index_to_create]
                recreate_dataset(subset_dataset, names, index_to_create, output_path, number_of_folds, datasets_settings)
        elif process == PROCESS_OPTIONS[4]:
            was_processed= input("Has the dataset been processed before? (y/n): ").lower() == "y"
            set_types = ["Full dataset", "Training dataset", "Testing dataset"]
            set_type = show_menu("What type of dataset do you want to calculate stats for? ", set_types)
            if set_type == set_types[0]:
                if was_processed:
                    dataset = load_meta_features_csv("")
                else:
                    dataset = prepare_meta_feature_dataset_for_states()
            else:
                if was_processed:
                    if set_type == set_types[1]:
                        dataset = load_meta_features_csv("training")
                    else:
                        dataset = load_meta_features_csv("testing")
                else:
                    training_set, testing_set = prepare_meta_feature_sets()
                    if set_type == set_types[1]:
                        dataset = training_set
                    else:
                        dataset = testing_set
            calculate_dataset_stats(dataset)
        elif process == PROCESS_OPTIONS[5]:
            has_sets= input("Do you have training sets? (y/n): ").lower() == "y"
            if has_sets:
                training_set = load_meta_features_csv("training")
            else:
                training_set, _ = prepare_meta_feature_sets()
            optimise_meta_learners(training_set)
        elif process == PROCESS_OPTIONS[6]:
            has_sets= input("Do you have training and testing sets? (y/n): ").lower() == "y"
            if has_sets:
                training_set = load_meta_features_csv("training")
                testing_set = load_meta_features_csv("testing")
            else:
                training_set, testing_set = prepare_meta_feature_sets()
            train_meta_learners(training_set, testing_set)
        elif process == PROCESS_OPTIONS[7]:
            calculate_meta_learners_stats()
        elif process == PROCESS_OPTIONS[8]:
            dataset_names = show_dataset_menu(datasets_settings)
            if not dataset_names:
                break

            meta_learners_results = load_results_csv()
            number_of_folds = int(input("How many folds do you want to use per instance? "))
            transformer_path = input("Enter the path of the pipeline file: ")
            output_path = input("Enter the path of the output dataset folder: ")
            results = pd.DataFrame()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"meta_learning_testing_results_{timestamp}.csv"
            file_path = output_path + "\\" + file_name
            for dataset_name in dataset_names:
                dataset_settings = next((item for item in datasets_settings if item["name"] == dataset_name), None)
                dataset_result = test_meta_learner(dataset_name,
                                                   dataset_settings,
                                                   meta_learners_results,
                                                   number_of_folds,
                                                   transformer_path)
                results = pd.concat([results, dataset_result], ignore_index=True)
                save_data_frame(results, file_path)
            print(results)
        else:
            break

# Using the special variable
# __name__
if __name__=="__main__":
    main()