
from datetime import datetime
import random
import pandas as pd

from ModelTrainer.decisionTreeTrainer import training_all_decision_trees
from ModelTrainer.knnTrainer import training_all_k_nearest_neighbors
from ModelTrainer.randomForestTrainer import training_all_random_forests
from ModelTrainer.svmTrainer import training_all_support_vector_machines
from ModelTrainer.nnTrainer import training_all_neural_networks
from Utils.fileHandler import save_data_frame
from Utils.menus import show_meta_leaner_type_menu


def train_meta_learners(training_dataset, testing_dataset):
    selected_meta_learn_types = show_meta_leaner_type_menu()
    number_of_folds = int(input("How many folds do you want to use per instance? "))
    results = pd.DataFrame(columns=["model type", "technique",  "training mses", "testing mses"])
    for selected_meta_learn_type in selected_meta_learn_types:
        settings_file_path = input(f"Enter the path of the {selected_meta_learn_type}' settings file: ")
        seed = random.randint(0, 4294967295)
        if selected_meta_learn_type == 'Decision trees':
            result = training_all_decision_trees(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == 'Random forests':
            result = training_all_random_forests(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == 'K-nearest neighbors':
            result = training_all_k_nearest_neighbors(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == 'Neural networks':
            result = training_all_neural_networks(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        else:
            result = training_all_support_vector_machines(settings_file_path , training_dataset, testing_dataset, seed,number_of_folds)
        results = pd.concat([results, pd.DataFrame(result)], ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName = f"Data/Datasets/Results/meta_learners_results_{timestamp}.csv"
    save_data_frame(results, fileName)