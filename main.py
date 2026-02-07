import pandas as pd
import torch

from InstanceCreator.instanceCreator import create_dataset, recreate_subsets, recreate_dataset
from ModelTrainer.metaLearnersTrainer import train_meta_learners
from Optimisers.metaLearnersOptimiser import optimise_meta_learners
from Optimisers.nnOptimiser import optimise_nn
from Utils.createAvgNNSetting import create_generic_nn_setting
from Utils.datasetStatsCalculator import calculate_dataset_stats, calculate_meta_learners_stats
from Utils.metaFeatureDatasetHandler import load_meta_feature_dataset
from Utils.menus import show_dataset_menu, show_menu, show_dataset_setting_menu

process_options = ["Optimise NN",  #0-1
                   "Create Avg NN Settings",  #1-2
                   "Create Subsets and instances",  #2-3
                   "Recreate Subsets",  #3-4
                   "Recreate instances",  #4-5
                   "Get Statistics of Meta Learning Dataset",  #5-6
                   "Optimise Meta Learning",  #6-7
                   "Train Meta Learning",  #7-8
                   "Get Statistics of Meta Learners results",  #8-9
                   "Exit"]

def main():
    print(f"PyTorch version: {torch.__version__}")  # Ensure it's a CUDA-compatible version
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version {torch.version.cuda}")
    else:
        print(f"Device: CPU")
    print(f"")
    while True:
        process = show_menu("Select process by entering a number: ", process_options)
        if process == process_options[0]:
            while True:
                datasets_settings = show_dataset_setting_menu()
                if not datasets_settings:
                    break
                names = show_dataset_menu(datasets_settings)
                if not names:
                    break
                for name in names:
                    dataset_settings = next((item for item in datasets_settings if item["name"] == name), None)
                    quited = optimise_nn(name, dataset_settings)
                    if quited:
                         break
        elif process == process_options[1]:
            create_generic_nn_setting()
        elif process == process_options[2]:
            while True:
                datasets_settings = show_dataset_setting_menu()
                if not datasets_settings:
                    break
                names = show_dataset_menu(datasets_settings)
                if not names:
                    break
                output_path = input("Enter the path of the Output dataset file or folder: ")
                settings_file_path = input("Enter the path of the NN's settings file: ")
                number_of_instances = int(input("How many Subsets do you want to create per dataset? "))
                number_of_folds = int(input("How many folds do you want to use per instance? "))
                for name in names:
                    dataset_settings = next((item for item in datasets_settings if item["name"] == name), None)
                    output_path = create_dataset(name,
                                                 output_path,
                                                 number_of_instances,
                                                 settings_file_path,
                                                 number_of_folds,
                                                 dataset_settings)
        elif process == process_options[3]:
            datasets_settings = show_dataset_setting_menu()
            if datasets_settings:
                if input("Do you have a meta-feature file? (y/n): ").lower() == "y":
                    dataset = load_meta_feature_dataset(True)
                    names =[]
                    number_of_instances = int(input("How many Subsets do you what to create per dataset? "))
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
                        number_of_instances = int(input("How many Subsets do you what to create per dataset? "))
                        recreate_subsets(dataset, number_of_instances, datasets_settings, names)
        elif process == process_options[4]:
            datasets_settings = show_dataset_setting_menu()
            if datasets_settings:
                names = show_dataset_menu(datasets_settings)
                if names:
                    subset_dataset = load_meta_feature_dataset(True)
                    output_path = input("Enter the path of the Output dataset file or folder: ")
                    settings_file_path =input("Enter the path of the NN's settings file: ")
                    number_of_folds = int(input("How many folds do you what use per instance? "))
                    index_to_create = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
                    recreate_dataset(subset_dataset, names, index_to_create, settings_file_path, output_path, number_of_folds, datasets_settings)
        elif process == process_options[5]:
            dataset = load_meta_feature_dataset()
            calculate_dataset_stats(dataset)
        elif process == process_options[6]:
            training_set = load_meta_feature_dataset(type = "training set", should_cover_to_binary = True)
            optimise_meta_learners(training_set)
        elif process == process_options[7]:
            training_set = load_meta_feature_dataset(type = "training set", should_cover_to_binary = True)
            testing_set = load_meta_feature_dataset(type = "testing set", should_cover_to_binary = True)
            train_meta_learners(training_set, testing_set)
        elif process == process_options[8]:
            calculate_meta_learners_stats()
        else:
            break

# Using the special variable
# __name__
if __name__=="__main__":
    main()