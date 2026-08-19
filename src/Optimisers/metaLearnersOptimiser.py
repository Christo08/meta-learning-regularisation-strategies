import json
from datetime import datetime

from src.Optimisers.decisionTreeOptimiser import optimise_decision_tree
from src.Optimisers.knnOptimiser import optimise_k_nearest_neighbors
from src.Optimisers.nnOptimiser import optimise_mate_nn
from src.Optimisers.randomForsetOptimiser import optimise_random_forest
from src.Optimisers.svmOptimiser import optimise_support_vector_machine
from src.Utils.constants import META_LEARNERS_SETTINGS_PATH, OPTIMED_METRIC_OPTIONS
from src.Utils.fileHandler import save_meta_learner_settings, ObjectEncoder
from src.Utils.menus import META_LEARN_TYPES, show_menu
from src.Utils.menus import show_meta_leaner_type_menu


def optimise_meta_learners(training_set, validation_set):
    training_set.drop(columns=["dataset_name","file_name"], inplace=True)
    validation_set.drop(columns=["dataset_name", "file_name"], inplace=True)
    selected_meta_learn_types = show_meta_leaner_type_menu()
    if len(selected_meta_learn_types) == 0:
        return
    selected_metric_type = show_menu("Select the metric which will be optimed by entering its number: ", OPTIMED_METRIC_OPTIONS)
    direction = "min" if selected_metric_type == OPTIMED_METRIC_OPTIONS[2] else "max"
    setting_indexes = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    path = f"{META_LEARNERS_SETTINGS_PATH}\\{selected_metric_type.replace(" ","_")}_settings_index_{timestamp}.json"
    for selected_meta_learn_type in selected_meta_learn_types:
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            setting = optimise_decision_tree(training_set, validation_set, selected_metric_type, direction)
            module_type = "DecisionTrees"
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            setting = optimise_k_nearest_neighbors(training_set, validation_set, selected_metric_type, direction)
            module_type = "KNearestNeighbors"
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            setting = optimise_mate_nn(training_set, validation_set, selected_metric_type, direction)
            module_type = "NeuralNetworks"
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            setting = optimise_random_forest(training_set, validation_set, selected_metric_type, direction)
            module_type = "RandomForest"
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            setting = optimise_support_vector_machine(training_set, validation_set, selected_metric_type, direction)
            module_type = "SupportVectorMachines"
        else:
            return
        setting_indexes[module_type] = save_meta_learner_settings(setting, module_type)
        with open(path, "w") as file:
            json.dump(setting_indexes, file, indent=4, cls=ObjectEncoder)
