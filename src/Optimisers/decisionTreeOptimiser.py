import random
from datetime import datetime

import pyhopper

from src.ModelTrainer.decisionTreeTrainer import train_meta_decision_tree
from src.Utils.constants import META_LEANER_TARGET_COLUMNS, CHECK_POINTS_PATH, OPTIMED_METRIC_OPTIONS
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import folder_maker

number_of_steps = 400
parameter_group = {
    "criterion": pyhopper.choice(["gini", "entropy", "log_loss"]),
    "splitter": pyhopper.choice(["best", "random"]),
    "max_depth": pyhopper.int(1, 200),
    "min_samples_split": pyhopper.int(2, 60),
    "min_samples_leaf": pyhopper.int(1, 60),
    "ccp_alpha": pyhopper.float(0.0, 0.5, "0.4f")
}
training_set = {}
validation_set = {}
metric_type = ""


def optimise_decision_tree(training_dataset, validation_dataset, selected_metric_type, direction):
    global training_set, validation_set, metric_type

    settings = {}
    metric_type = selected_metric_type

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for target_column in META_LEANER_TARGET_COLUMNS:
        training_set = prepared_meta_feature_dataset(training_dataset, target_column, False)
        validation_set = prepared_meta_feature_dataset(validation_dataset, target_column, False)

        search = pyhopper.Search(parameter_group)
        check_point_path = f"{CHECK_POINTS_PATH}Meta-learners\\DecisionTree"
        folder_maker(check_point_path)
        best_params = search.run(
            train_decision_tree_warp,
            direction=direction,
            steps=number_of_steps,
            checkpoint_path=f"{check_point_path}\\{target_column}_{timestamp}"
        )
        validation_loses = train_decision_tree_warp(best_params)
        print(
        f"Tuned params for decision tree for {target_column} resulting in {validation_loses} {metric_type}")
        settings[target_column] = best_params
    return settings

def train_decision_tree_warp(params):
    global training_set, validation_set, metric_type
    seed = random.randint(0, 4294967295)
    stats, _ = train_meta_decision_tree(params, training_set, validation_set, seed, kFold=5, metric_type=metric_type)
    testing_stats = stats.get_best_testing_stats_json_object()
    if metric_type == OPTIMED_METRIC_OPTIONS[0]:
        return testing_stats["testing accuracies"]
    elif metric_type == OPTIMED_METRIC_OPTIONS[1]:
        return testing_stats["testing f1"]
    elif metric_type == OPTIMED_METRIC_OPTIONS[2]:
        return testing_stats["testing loses"]
    else:
        return testing_stats["testing precision"]