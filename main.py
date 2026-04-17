from datetime import datetime

import pandas as pd
import torch

from src.ModelTrainer.metaLearnersTrainer import train_meta_learners, test_meta_learner
from src.Optimisers.metaLearnersOptimiser import optimise_meta_learners
from src.Optimisers.nnOptimiser import optimise_basic_nn
from src.Utils.constants import *
from src.Utils.datasetSettingHandler import DatasetsSettingsHandler
from src.Utils.fileHandler import save_data_frame, load_settings, load_meta_features_csv, load_results_csv
from src.Utils.instanceCreator import create_dataset, recreate_subsets, recreate_dataset
from src.Utils.menus import show_menu, show_dataset_loader_menu
from src.Utils.statsCalculator import calculate_meta_learners_stats, calculate_dataset_stats


def main():
    print(f"PyTorch version: {torch.__version__}")  # Ensure it's a CUDA-compatible version
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version {torch.version.cuda}")
    else:
        print(f"Device: CPU")
    print(f"")
    datasets_settings_handler = DatasetsSettingsHandler()

    while True:
        process = show_menu("Select process by entering a number: ", PROCESS_OPTIONS)
        if process == PROCESS_OPTIONS[0]:
            while True:
                datasets_settings = datasets_settings_handler.select_datasets_settings()
                if not datasets_settings:
                    break
                parameter_group = show_menu("Select parameter group by entering a number:", PARAMETER_GROUPS)
                if parameter_group == PARAMETER_GROUPS[len(PARAMETER_GROUPS) - 1]:
                    break
                basic_settings = None
                if not(parameter_group == PARAMETER_GROUPS[0] or parameter_group == PARAMETER_GROUPS[1]):
                    basic_settings = load_settings(input("Enter the path to the basic settings file of the NN:"))
                for dataset_settings in datasets_settings:
                    optimise_basic_nn(dataset_settings, parameter_group, basic_settings)
        elif process == PROCESS_OPTIONS[1]:
            while True:
                datasets_settings = datasets_settings_handler.select_datasets_settings()
                if not datasets_settings:
                    break
                output_path = input("Enter the path of the output dataset file or folder: ")
                number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
                number_of_folds = int(input("How many folds do you want to use per instance? "))
                for dataset_settings in datasets_settings:
                    output_path = create_dataset(output_path, number_of_instances, number_of_folds, dataset_settings)
        elif process == PROCESS_OPTIONS[2]:
            if input("Do you have a meta-feature file? (y/n): ").lower() == "y":
                dataset = load_meta_features_csv()
                number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
                recreate_subsets(dataset, number_of_instances)
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
                names = datasets_settings_handler.select_dataset_name()
                if names:
                    number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
                    recreate_subsets(dataset, number_of_instances, names)
        elif process == PROCESS_OPTIONS[3]:
            names = datasets_settings_handler.select_dataset_name()
            if names:
                subset_dataset = load_meta_features_csv()
                output_path = input("Enter the path of the output dataset file or folder: ")
                number_of_folds = int(input("How many folds do you want to use per instance? "))
                index_to_create = input("Enter the indexes to recreate (separated by commas): ").replace(' ', '').split(",")
                index_to_create = [int(index) for index in index_to_create]
                recreate_dataset(subset_dataset, names, index_to_create, output_path, number_of_folds)
        elif process == PROCESS_OPTIONS[4]:
            dataset = show_dataset_loader_menu(allow_full_dataset = True)
            calculate_dataset_stats(dataset)
        elif process == PROCESS_OPTIONS[5]:
            training_set = show_dataset_loader_menu()
            optimise_meta_learners(training_set)
        elif process == PROCESS_OPTIONS[6]:
            training_set, testing_set = show_dataset_loader_menu(return_both_sets = True)
            train_meta_learners(training_set, testing_set)
        elif process == PROCESS_OPTIONS[7]:
            calculate_meta_learners_stats()
        elif process == PROCESS_OPTIONS[8]:
            datasets_settings = datasets_settings_handler.select_datasets_settings()
            if not datasets_settings:
                break

            meta_learners_results = load_results_csv()
            number_of_folds = int(input("How many folds do you want to use per instance? "))
            transformer_path = input("Enter the path of the pipeline file: ")
            output_path = input("Enter the path of the output dataset folder: ")
            results = pd.DataFrame()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"meta_learning_testing_results_{timestamp}.csv"
            file_path = output_path + "\\" + file_name
            for dataset_settings in datasets_settings:
                dataset_result = test_meta_learner(dataset_settings,
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