from src.Optimisers.decisionTreeOptimiser import optimise_decision_tree
from src.Optimisers.knnOptimiser import optimise_k_nearest_neighbors
from src.Optimisers.nnOptimiser import optimise_meta_leaner_nn
from src.Optimisers.randomForsetOptimiser import optimise_random_forest
from src.Optimisers.svmOptimiser import optimise_support_vector_machine
from src.Utils.fileHandler import save_meta_learner_settings
from src.Utils.menus import show_meta_leaner_type_menu
from src.Utils.menus import META_LEARN_TYPES


def optimise_meta_learners(dataset):
    selected_meta_learn_types = show_meta_leaner_type_menu()
    for selected_meta_learn_type in selected_meta_learn_types:
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            setting = optimise_decision_tree(dataset)
            save_meta_learner_settings(setting, "DecisionTrees")
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            setting = optimise_k_nearest_neighbors(dataset)
            save_meta_learner_settings(setting, "KNearestNeighbors")
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            setting = optimise_meta_leaner_nn(dataset)
            save_meta_learner_settings(setting, "NeuralNetworks")
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            setting = optimise_random_forest(dataset)
            save_meta_learner_settings(setting, "RandomForest")
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            setting = optimise_support_vector_machine(dataset)
            save_meta_learner_settings(setting, "SupportVectorMachines")
        else:
            return