import pandas as pd
import torch

from src.ModelTrainer.metaLearnersTrainer import train_meta_learners
from src.Optimisers.metaLearnersOptimiser import optimise_meta_learners
from src.Optimisers.nnOptimiser import optimise_basic_nn
from src.Utils.constants import *
from src.Utils.statsCalculator import calculate_meta_learners_stats, calculate_dataset_stats
from src.Utils.fileHandler import load_json_file, load_settings
from src.Utils.instanceCreator import create_dataset, recreate_subsets, recreate_dataset
from src.Utils.menus import show_dataset_menu, show_menu
from src.Utils.metaFeatureDatasetHandler import load_meta_feature_dataset, split_dataset


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
                output_path = input("Enter the path of the Output dataset file or folder: ")
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
                dataset = load_meta_feature_dataset(need_subsets_info = True, should_ask_for_apply_z_scoring = False, should_ask_rank_techniques = False)
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
                                                "SMOTE_training_loss","SMOTE_testing_loss","prune_training_loss","prune_testing_loss",
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
                subset_dataset = load_meta_feature_dataset(need_subsets_info=True, should_ask_for_apply_z_scoring = False, should_ask_rank_techniques = False)
                output_path = input("Enter the path of the Output dataset file or folder: ")
                number_of_folds = int(input("How many folds do you want to use per instance? "))
                index_to_create = input("Enter the indexes to recreate (separated by commas): ").replace(' ', '').split(",")
                index_to_create = [int(index) for index in index_to_create]
                recreate_dataset(subset_dataset, names, index_to_create, output_path, number_of_folds, datasets_settings)
        elif process == PROCESS_OPTIONS[4]:
            dataset = load_meta_feature_dataset(should_ask_for_apply_z_scoring = False)
            calculate_dataset_stats(dataset)
        elif process == PROCESS_OPTIONS[5]:
            dataset = load_meta_feature_dataset(should_ask_for_apply_z_scoring = True)
            split_dataset(dataset)
        elif process == PROCESS_OPTIONS[6]:
            # should_add_params = input("Do you to add the NN's meta-features to the dataset? (y/n): ").lower() == "y"
            training_set = load_meta_feature_dataset(type = "training set", should_cover_to_binary = True, should_ask_rank_techniques = False, should_ask_for_apply_z_scoring=True, should_add_params=False)
            optimise_meta_learners(training_set)
        elif process == PROCESS_OPTIONS[7]:
            # should_add_params = input("Do you to add the NN's meta-features to the datasets? (y/n): ").lower() == "y"
            training_set = load_meta_feature_dataset(type = "training set", should_cover_to_binary = True, should_ask_rank_techniques = False, should_ask_for_apply_z_scoring=True, should_add_params=False)
            testing_set = load_meta_feature_dataset(type = "testing set", should_cover_to_binary = True, should_ask_rank_techniques = False, should_ask_for_apply_z_scoring=True, should_add_params=False)
            train_meta_learners(training_set, testing_set)
        elif process == PROCESS_OPTIONS[8]:
            calculate_meta_learners_stats()
        else:
            break

# Using the special variable
# __name__
if __name__=="__main__":
    main()