import random
from datetime import datetime

import pandas as pd

from src.ModelTrainer.decisionTreeTrainer import training_meta_decision_trees
from src.ModelTrainer.knnTrainer import training_meta_k_nearest_neighbors
from src.ModelTrainer.randomForestTrainer import training_meta_random_forests
from src.ModelTrainer.svmTrainer import training_meta_support_vector_machines
from src.Utils.constants import META_LEARN_TYPES
from src.Utils.fileHandler import save_data_frame, folder_maker, load_json_file
from src.Utils.menus import show_meta_leaner_type_menu


def train_meta_learners(training_dataset, testing_dataset):
    selected_meta_learn_types = show_meta_leaner_type_menu()
    number_of_folds = int(input("How many folds do you want to use per instance? "))
    results = pd.DataFrame(columns=["model type", "technique",  "training loses", "testing loses"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings_file_path = input(f"Enter the path of the settings index file: ")
    output_path = input("Enter the path of the Output stats folder: ")
    settings = load_json_file(settings_file_path)

    for selected_meta_learn_type in selected_meta_learn_types:
        settings_file_path = settings[selected_meta_learn_type]
        seed = random.randint(0, 4294967295)
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            result = training_meta_decision_trees(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            result = training_meta_random_forests(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            result = training_meta_k_nearest_neighbors(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            print(1)
            # result = training_meta_nns(settings_file_path, training_dataset, testing_dataset, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            result = training_meta_support_vector_machines(settings_file_path, training_dataset, testing_dataset, seed, number_of_folds)
        else:
            return
        results = pd.concat([results, pd.DataFrame(result)], ignore_index=True)
    folder_maker(output_path)
    file_name = f"{output_path}meta_learners_results_{timestamp}.csv"
    save_data_frame(results, file_name)