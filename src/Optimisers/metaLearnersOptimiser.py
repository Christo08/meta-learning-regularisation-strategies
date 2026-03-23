import json
from datetime import datetime

from src.Optimisers.decisionTreeOptimiser import optimise_decision_tree
from src.Optimisers.knnOptimiser import optimise_k_nearest_neighbors
from src.Optimisers.nnOptimiser import optimise_meta_leaner_nn
from src.Optimisers.randomForsetOptimiser import optimise_random_forest
from src.Optimisers.svmOptimiser import optimise_support_vector_machine
from src.Utils.constants import META_LEARNERS_SETTINGS_PATH
from src.Utils.fileHandler import save_meta_learner_settings, ObjectEncoder
from src.Utils.menus import META_LEARN_TYPES
from src.Utils.menus import show_meta_leaner_type_menu


def optimise_meta_learners(dataset):
    selected_meta_learn_types = show_meta_leaner_type_menu()
    setting_indexes = {}
    for selected_meta_learn_type in selected_meta_learn_types:
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            setting = optimise_decision_tree(dataset)
            module_type = "DecisionTrees"
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            setting = optimise_k_nearest_neighbors(dataset)
            module_type = "KNearestNeighbors"
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            setting = optimise_meta_leaner_nn(dataset)
            module_type = "NeuralNetworks"
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            setting = optimise_random_forest(dataset)
            module_type = "RandomForest"
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            setting = optimise_support_vector_machine(dataset)
            module_type = "SupportVectorMachines"
        else:
            return
        setting_indexes[module_type] = save_meta_learner_settings(setting, module_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{META_LEARNERS_SETTINGS_PATH}\\settings_index_{timestamp}.json"
    with open(path, "x") as file:
        json.dump(setting_indexes, file, indent=4, cls=ObjectEncoder)
