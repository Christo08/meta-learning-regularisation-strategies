from Optimisers.decisionTreeOptimiser import optimise_decision_tree
from Optimisers.knnOptimiser import optimise_k_nearest_neighbors
from Optimisers.nnOptimiser import optimise_meta_leaner_nn
from Optimisers.randomForsetOptimiser import optimise_random_forest
from Optimisers.svmOptimiser import optimise_support_vector_machine
from Utils.fileHandler import save_meta_learner_settings
from Utils.menus import show_meta_leaner_type_menu
from Utils.menus import META_LEARN_TYPES


def optimise_meta_learners(dataset):
    selected_meta_learn_types = show_meta_leaner_type_menu()
    output_path = input("Enter the path to the folder where the settings should be saved: ")
    for selected_meta_learn_type in selected_meta_learn_types:
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            setting = optimise_decision_tree(dataset)
            save_meta_learner_settings(setting, output_path, "DecisionTrees")
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            setting = optimise_k_nearest_neighbors(dataset)
            save_meta_learner_settings(setting, output_path, "KNearestNeighbors")
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            setting = optimise_meta_leaner_nn(dataset)
            save_meta_learner_settings(setting, output_path, "NeuralNetworks")
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            setting = optimise_random_forest(dataset)
            save_meta_learner_settings(setting, output_path, "RandomForest")
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            setting = optimise_support_vector_machine(dataset)
            save_meta_learner_settings(setting, output_path, "SupportVectorMachines")
        else:
            return