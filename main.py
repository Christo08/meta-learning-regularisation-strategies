import random
from datetime import datetime

import torch

from src.ModelTrainer.metaLearnersTrainer import train_meta_learners, test_meta_learner
from src.Models.metaLearner import MetaLearner
from src.Optimisers.metaLearnersOptimiser import optimise_meta_learners
from src.Optimisers.nnOptimiser import optimise_basic_nn
from src.Utils.constants import *
from src.Utils.datasetSettingHandler import DatasetsSettingsHandler
from src.Utils.fileHandler import load_settings, load_meta_features_csv, load_results_csv
from src.Utils.instanceCreator import create_subsets, create_dataset_for_subset, recreate_meta_features
from src.Utils.menus import show_menu, show_dataset_loader_menu
from src.Utils.metaFeatureDatasetHandler import prepare_meta_feature_sets
from src.Utils.statsCalculator import calculate_meta_learners_stats, calculate_dataset_stats, \
    calculate_meta_learners_performance


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
            datasets_settings = datasets_settings_handler.select_datasets_settings()
            if not datasets_settings:
                break
            output_path = input("Enter a path to the folder where the subsets index will be saved: ")
            number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
            has_seeds = input("Do you have starting seeds (Y/N)?") == "Y"
            seeds = []
            if has_seeds:
                print(f'Please enter {len(datasets_settings)} seeds:')
                for _ in range(len(datasets_settings)):
                    seeds.append(float(input()))
            else:
                for _ in range(len(datasets_settings)):
                    seeds.append(random.randint(0, 4294967295))
            for dataset_settings, seed in zip(datasets_settings, seeds):
                output_path = create_subsets(output_path, number_of_instances, dataset_settings, seed)
        elif process == PROCESS_OPTIONS[2]:
            while True:
                datasets_settings = datasets_settings_handler.select_datasets_settings()
                if not datasets_settings:
                    break
                has_indexes = input("Do you have specific indexes that you want to recreate (Y/N)?") == "Y"
                indexes = None
                if has_indexes:
                    text_indexes = input("Enter the indexes of the subset you would like to create (separated by a comma): ")
                    indexes = [int(index) for index in text_indexes.split(',')]
                output_path = input("Enter the path to the subsets index and where the output will be saved: ")
                number_of_folds = 10
                create_dataset_for_subset(output_path, number_of_folds, datasets_settings, indexes)
        elif process == PROCESS_OPTIONS[3]:
            while True:
                datasets_settings = datasets_settings_handler.select_datasets_settings()
                if not datasets_settings:
                    break
                has_indexes = input("Do you have specific indexes that you want to recreate (Y/N)?") == "Y"
                indexes = None
                if has_indexes:
                    text_indexes = input("Enter the indexes of the subset you would like to create (separated by a comma): ")
                    indexes = [int(index) for index in text_indexes.split(',')]
                output_path = input("Enter the path to the subsets index and where the output will be saved: ")
                number_of_folds = 10
                recreate_meta_features(output_path, number_of_folds, datasets_settings, indexes)
        elif process == PROCESS_OPTIONS[4]:
            dataset = show_dataset_loader_menu(allow_full_dataset = True)
            calculate_dataset_stats(dataset)
        elif process == PROCESS_OPTIONS[5]:
            training_set, validation_set = show_dataset_loader_menu(return_both_sets = True)
            optimise_meta_learners(training_set, validation_set)
        elif process == PROCESS_OPTIONS[6]:
            training_set, validation_set = show_dataset_loader_menu(return_both_sets = True)
            train_meta_learners(training_set, validation_set)
        elif process == PROCESS_OPTIONS[7]:
            calculate_meta_learners_stats()
        elif process == PROCESS_OPTIONS[8]:
            meta_learners_results = load_results_csv()
            output_path = input("Enter the path of the output dataset folder: ")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if input("Do you have validation sets? (y/n): ").lower() == "y":
                validation_set = load_meta_features_csv("testing")
            else:
                validation_set = prepare_meta_feature_sets()[1]
            if input("Do you have a ranked validation sets? (y/n): ").lower() == "y":
                ranked_validation_set = load_meta_features_csv("testing")
            else:
                ranked_validation_set = prepare_meta_feature_sets()[1]
            file_name = f"meta_learning_testing_results_{timestamp}.csv"
            file_path = output_path + "\\" + file_name
            meta_learner = MetaLearner(meta_learners_results)
            test_meta_learner(validation_set, ranked_validation_set, file_path, meta_learner)
        elif process == PROCESS_OPTIONS[9]:
            calculate_meta_learners_performance()
        else:
            break

# Using the special variable
# __name__
if __name__=="__main__":
    main()