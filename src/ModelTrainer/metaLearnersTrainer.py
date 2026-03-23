import os
from datetime import datetime
import random
import pandas as pd

from src.ModelTrainer.decisionTreeTrainer import training_all_decision_trees
from src.ModelTrainer.knnTrainer import training_all_k_nearest_neighbors
from src.ModelTrainer.randomForestTrainer import training_all_random_forests
from src.ModelTrainer.svmTrainer import training_all_support_vector_machines
from src.ModelTrainer.nnTrainer import training_all_neural_networks
from src.Utils.constants import META_LEARN_TYPES, RESULTS_PATH
from src.Utils.fileHandler import save_data_frame
from src.Utils.menus import show_meta_leaner_type_menu


def train_meta_learners(training_dataset, testing_dataset):
    selected_meta_learn_types = show_meta_leaner_type_menu()
    number_of_folds = int(input("How many folds do you want to use per instance? "))
    results = pd.DataFrame(columns=["model type", "technique",  "training loses", "testing loses"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = {}
    for selected_meta_learn_type in selected_meta_learn_types:
        settings_file_path = input(f"Enter the path of the {selected_meta_learn_type}' settings file: ")
        settings[selected_meta_learn_type] = settings_file_path
    for selected_meta_learn_type in selected_meta_learn_types:
        settings_file_path = settings[selected_meta_learn_type]
        seed = random.randint(0, 4294967295)
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            result = training_all_decision_trees(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            result = training_all_random_forests(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            result = training_all_k_nearest_neighbors(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            result = training_all_neural_networks(settings_file_path , training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            result = training_all_support_vector_machines(settings_file_path , training_dataset, testing_dataset, seed,number_of_folds)
        else:
            return
        results = pd.concat([results, pd.DataFrame(result)], ignore_index=True)
    if not os.path.isdir(RESULTS_PATH):
        os.makedirs(RESULTS_PATH, exist_ok=True)
    file_name = f"{RESULTS_PATH}meta_learners_results_{timestamp}.csv"
    save_data_frame(results, file_name)