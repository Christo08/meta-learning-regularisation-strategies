from Optimisers.decisionTreeOptimiser import optimise_decision_tree
from Optimisers.knnOptimiser import optimise_k_nearest_neighbors
from Optimisers.nnOptimiser import optimise_meta_leaner_nn
from Optimisers.randomForsetOptimiser import optimise_random_forest
from Optimisers.svmOptimiser import optimise_support_vector_machine
from Utils.fileHandler import save_meta_learner_settings
from Utils.menus import show_meta_leaner_type_menu


def optimise_meta_learners(dataset):
    selected_meta_learn_types = show_meta_leaner_type_menu()
    for selected_meta_learn_type in selected_meta_learn_types:
        if selected_meta_learn_type == 'Decision trees':
            setting = optimise_decision_tree(dataset)
            save_meta_learner_settings(setting, "DecisionTrees")
        elif selected_meta_learn_type == 'Random forests':
            setting = optimise_random_forest(dataset)
            save_meta_learner_settings(setting, "RandomForest")
        elif selected_meta_learn_type == 'K-nearest neighbors':
            setting = optimise_k_nearest_neighbors(dataset)
            save_meta_learner_settings(setting, "KNearestNeighbors")
        elif selected_meta_learn_type == 'Support vector machines':
            setting = optimise_support_vector_machine(dataset)
            save_meta_learner_settings(setting, "SupportVectorMachines")
        elif selected_meta_learn_type == 'Neural networks':
            setting = optimise_meta_leaner_nn(dataset)
            save_meta_learner_settings(setting, "NeuralNetworks")
        else:
            return