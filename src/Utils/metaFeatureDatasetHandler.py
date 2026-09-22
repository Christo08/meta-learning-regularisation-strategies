import ast
import math
import random
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.stats import ttest_ind, friedmanchisquare
from sympy import true, false

try:
    from scipy.stats import studentized_range
except ImportError:
    studentized_range = None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.Utils.constants import TARGET_COLUMNS, TEST_TYPES
from src.Utils.fileHandler import load_meta_features_csv, save_data_frame
from src.Utils.menus import show_menu


def spilt_dataset_and_targets(dataset, target_columns=TARGET_COLUMNS):
    missing = False
    for target_column in target_columns:
        if target_column not in dataset.columns:
            missing = True
            break
    if missing:
        targets = pd.DataFrame()
        return dataset, targets
    else:
        targets = dataset[target_columns]
        dataset = dataset.drop(target_columns, axis=1)
        return dataset, targets

def split_dataset(dataset):
    targets = [col for col in TARGET_COLUMNS if col != "SMOTE"]
    selected_columns = ['dataset_name'] + targets
    subset = dataset[selected_columns]
    rankings_per_dataset = subset.groupby('dataset_name')[targets].apply(lambda x: (x == 1).sum()).reset_index()

    rankings_per_dataset["mean_loss"] = rankings_per_dataset[targets].mean(axis=1)
    rankings_per_dataset["bin"] = pd.qcut(rankings_per_dataset["mean_loss"], q=4, labels=False)
    #Seed use to recreate meta-learner 5232 9760
    seed = 9851
    train, test = train_test_split(rankings_per_dataset,
                                   test_size=0.25,
                                   stratify=rankings_per_dataset["bin"],
                                   random_state = seed)

    train_datasets_name = train["dataset_name"].tolist()
    train_datasets_name.remove("Page Blocks")
    train_datasets_name.append("Online shopper")
    test_datasets_name = test["dataset_name"].tolist()
    test_datasets_name.remove("Online shopper")
    test_datasets_name.append("Page Blocks")
    print("Seed:", seed)
    print("Train datasets:")
    print(train_datasets_name)

    print("Test datasets:")
    print(test_datasets_name)
    training_set = dataset[dataset["dataset_name"].isin(train_datasets_name)]
    testing_set = dataset[dataset["dataset_name"].isin(test_datasets_name)]

    return training_set, testing_set, seed

def remove_hyperparameters(dataset):
    hyperparameter_columns = [
        "batch_size",
        "learning_rate",
        "number_of_epochs",
        "number_of_hidden_layers",
        "avg_number_of_neurons",
        "min_number_of_neurons",
        "max_number_of_neurons",
        "total_number_of_neurons"
    ]

    return dataset.drop(columns=hyperparameter_columns, errors='ignore')

def remove_meta_features(dataset):
    meta_features_columns = [
        "number_of_features",
        "proportion_of_numeric_features",
        "number_of_instances",
        "number_of_classes",
        "ratio_of_instances_to_features",
        "ratio_of_classes_to_features",
        "ratio_of_instances_to_classes",
        "ratio_of_min_to_max_instances_per_class",
        "proportion_of_features_with_outliers",
        "average_mutual_information",
        "minimum_mutual_information",
        "maximum_mutual_information",
        "equivalent_number_of_features",
        "noise_to_signal_ratio_of_features"
    ]

    return dataset.drop(columns=meta_features_columns, errors='ignore')

def add_hyperparameters(dataset, settings):
    dataset["batch_size"] = settings["batch_size"]
    dataset["learning_rate"] = settings["learning_rate"]
    dataset["number_of_epochs"] = settings["number_of_epochs"]
    dataset["number_of_hidden_layers"] = settings["number_of_hidden_layers"]
    number_of_neurons = settings["number_of_neurons_in_layers"][:settings["number_of_hidden_layers"]]
    dataset["avg_number_of_neurons"] = np.average(number_of_neurons)
    dataset["min_number_of_neurons"] = np.min(number_of_neurons)
    dataset["max_number_of_neurons"] = np.max(number_of_neurons)
    dataset["total_number_of_neurons"] = np.sum(number_of_neurons)
    return dataset


def prepare_meta_feature_dataset_for_states():
    dataset = load_meta_features_csv()

    options = ""

    dataset = clean_dataset(dataset, False)

    targets = dataset[TARGET_COLUMNS]
    features = dataset.drop(TARGET_COLUMNS, axis=1)

    test_type = show_menu("Select the test method which will be used to rank the techniques: ", TEST_TYPES)
    if test_type == TEST_TYPES[0]:
        targets = rank_techniques_mann_whitney(targets)
    else:
        targets = rank_techniques_friedman(targets)
    should_cover_to_binary = input("Do you want to convert the ranks to binary (1 for best technique, 0 for others)? (y/n): ").lower() == "y"
    if should_cover_to_binary:
        options = options+"binary_"
        for column in TARGET_COLUMNS:
            if column in targets.columns:
                targets[column] = targets[column].apply(lambda x: 1 if x == 1 else 0)

    should_remove_hyperparameters = input("Do you want to remove hyperparameters (y/n): ").lower() == "y"
    if should_remove_hyperparameters:
        options = "removed_hyperparameters_"
        features = remove_hyperparameters(features)
    else:
        should_remove_meta_features = input("Do you want to remove meta features? (y/n): ").lower() == "y"
        if should_remove_meta_features:
            options = "removed_meta_features_"
            features = remove_meta_features(features)

    ignore_columns = ["dataset_name", "file_name"]

    should_apply_transformers = input("Do you want to apply transformers? (y/n): ").lower() == "y"
    transformer = None
    if should_apply_transformers:
        options = options+"transformers_"
        features, transformer = apply_transformers(features = features)

    scaler = None
    should_normalise= input("Do you want to apply z-scoring? (y/n): ").lower() == "y"
    if should_normalise:
        options = options+"z-scoring_"
        features, scaler = apply_normalization(features=features, ignore_columns = ignore_columns)

    dataset = pd.concat([features, targets], axis=1)

    should_save_dataset = input("Do you want to save the prepared dataset? (y/n): ").lower() == "y"
    if should_save_dataset:
        output_path = input("Enter the path of the output dataset folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"regularisation_meta_learning_{options}{timestamp}.csv"
        file_path = output_path + "\\" + file_name
        save_data_frame(dataset, file_path)
        dump({"transformer": transformer, "scaler": scaler}, f"Models/Settings/DataPipeline/pipeline_{options}{timestamp}.joblib")

    return dataset

def prepare_meta_feature_full_dataset_for_states(meta_features, path_to_data_pipeline):
    obj = load(path_to_data_pipeline)
    transformer = obj["transformer"]
    scaler = obj["scaler"]

    meta_features = clean_dataset(meta_features, True)
    meta_features, _ = apply_transformers(features = meta_features, transformer = transformer)
    # meta_features = create_bins(features = meta_features)
    # ignore_columns = ['proportion_of_numeric_features', 'minimum_mutual_information']
    meta_features, _ = apply_normalization(features = meta_features, ignore_columns = [], scaler = scaler)
    return meta_features


def prepare_meta_feature_sets():
    dataset = load_meta_features_csv()
    dataset = clean_dataset(dataset, False)

    targets = dataset[TARGET_COLUMNS]
    features = dataset.drop(TARGET_COLUMNS, axis=1)
    test_type = show_menu("Select the test method which will be used to rank the techniques: ", TEST_TYPES)
    if test_type == TEST_TYPES[0]:
        targets = rank_techniques_mann_whitney(targets)
    else:
        targets = rank_techniques_friedman(targets)
    dataset = pd.concat([features, targets], axis=1)

    training_set, validation_set, seed = split_dataset(dataset)

    should_cover_to_binary = input("Do you want to convert the ranks to binary (1 for best technique, 0 for others)? (y/n): ").lower() == "y"
    if should_cover_to_binary:
        for column in TARGET_COLUMNS:
            if column in training_set.columns:
                training_set[column] = training_set[column].apply(lambda x: 1 if x == 1 else 0)
                validation_set[column] = validation_set[column].apply(lambda x: 1 if x == 1 else 0)

    options = ""
    ignore_columns = ["dataset_name","file_name"]

    should_remove_hyperparameters = input("Do you want to remove hyperparameters (y/n): ").lower() == "y"
    if should_remove_hyperparameters:
        options = "removed_hyperparameters_"
        training_set = remove_hyperparameters(training_set)
        validation_set = remove_hyperparameters(validation_set)
    else:
        should_remove_meta_features = input("Do you want to remove meta features? (y/n): ").lower() == "y"
        if should_remove_meta_features:
            options = "removed_meta_features_"
            training_set = remove_meta_features(training_set)
            validation_set = remove_meta_features(validation_set)

    training_targets = training_set[TARGET_COLUMNS]
    training_features = training_set.drop(TARGET_COLUMNS, axis=1)

    validation_targets = validation_set[TARGET_COLUMNS]
    validation_features = validation_set.drop(TARGET_COLUMNS, axis=1)

    should_apply_transformers = input("Do you want to apply transformers? (y/n): ").lower() == "y"
    transformer = None
    if should_apply_transformers:
        options = options+"transformers_"
        training_features, validation_features, transformer = apply_transformers(training_features = training_features, testing_features = validation_features)

    should_apply_z_scoring = input("Do you want to apply z-scoring? (y/n): ").lower() == "y"
    scaler = None
    if should_apply_z_scoring:
        options = options+"z_scoring_"
        training_features, validation_features, scaler = apply_normalization(training_features = training_features, testing_features = validation_features, ignore_columns = ignore_columns)

    training_set = pd.concat([training_features, training_targets], axis=1)
    validation_set = pd.concat([validation_features, validation_targets], axis=1)

    should_save_dataset = input("Do you want to save the prepared dataset? (y/n): ").lower() == "y"
    if should_save_dataset:
        output_path = input("Enter the path of the output dataset folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"validation_set_{options}{timestamp}_{seed}.csv"
        file_path = output_path + "\\ValidationSets\\" + file_name
        save_data_frame(validation_set, file_path)
        file_name = f"training_set_{options}{timestamp}_{seed}.csv"
        file_path = output_path + "\\TrainingSets\\" + file_name
        save_data_frame(training_set, file_path)
        dump({"transformer": transformer, "scaler": scaler}, f"Models/Settings/DataPipeline/pipeline_{options}{timestamp}_{seed}.joblib")

    return training_set, validation_set

def clean_dataset(dataset, should_drop_dataset_name = True):
    columns_to_drop = ["best_training_technique",
                       "best_validation_technique",
                       "best_testing_technique",
                       "seed",
                       "subset_type"]
    columns_to_drop = columns_to_drop + [col for col in dataset.columns if '_training_loss' in col]
    columns_to_drop = columns_to_drop + [col for col in dataset.columns if 'SMOTE' in col]
    columns_to_drop = columns_to_drop + [col for col in dataset.columns if '_validation_loss' in col]
    columns_to_drop = columns_to_drop + [col for col in dataset.columns if '_f1_scores' in col]
    columns_to_drop = columns_to_drop + [col for col in dataset.columns if '_accuracies' in col]
    if should_drop_dataset_name:
        columns_to_drop = columns_to_drop + ["dataset_name"]

    dataset.drop(columns=columns_to_drop, errors="ignore", inplace=True)

    suffix = "_testing_loss"
    rename_map = {
        col: col[:-len(suffix)]
        for col in dataset.columns
        if isinstance(col, str) and col.endswith(suffix)
    }
    dataset.rename(columns=rename_map, inplace=True)

    for column in dataset.columns:
        if not(column in TARGET_COLUMNS) and not (column in ["dataset_name", "seed", "file_name", "subset_type"]):
            dataset[column] = dataset[column].values.astype(np.float64)

    return dataset

# def create_bins(features=None, training_features=None, testing_features=None):
#     if features is not None:
#         features = features.copy()
#         #convert proportion_of_numeric_features to multi classes
#         if 'proportion_of_numeric_features' in features.columns:
#             features['proportion_of_numeric_features'] = pd.cut(
#                 features['proportion_of_numeric_features'],
#                 bins=[0, 0.5, 0.8, 0.95, 1.0],
#                 labels=False,
#                 include_lowest=True
#             )
#         #convert minimum_mutual_information to binary classes where 1 is that minimum_mutual_information == 0
#         if 'minimum_mutual_information' in features:
#             features['minimum_mutual_information'] = (features['minimum_mutual_information'] == 0).astype(int)
#         return features
#     elif training_features is not None and testing_features is not None:
#         training_features = training_features.copy()
#         testing_features = testing_features.copy()
#
#         #convert proportion_of_numeric_features to multi classes
#         if 'proportion_of_numeric_features' in training_features.columns:
#             training_features['proportion_of_numeric_features'] = pd.cut(
#                 training_features['proportion_of_numeric_features'],
#                 bins=[0, 0.5, 0.8, 0.95, 1.0],
#                 labels=False,
#                 include_lowest=True
#             )
#             testing_features['proportion_of_numeric_features'] = pd.cut(
#                 testing_features['proportion_of_numeric_features'],
#                 bins=[0, 0.5, 0.8, 0.95, 1.0],
#                 labels=False,
#                 include_lowest=True
#             )
#         #convert minimum_mutual_information to binary classes where 1 is that minimum_mutual_information == 0
#         if 'minimum_mutual_information' in training_features:
#             training_features['minimum_mutual_information'] = (training_features['minimum_mutual_information'] == 0).astype(int)
#             testing_features['minimum_mutual_information'] = (testing_features['minimum_mutual_information'] == 0).astype(int)
#
#         return training_features, testing_features
#     else:
#         assert False, "Either dataset or both training_set and testing_set must be provided."

def apply_transformers(features = None, training_features = None, testing_features = None, transformer = None):
        if transformer == None:
            yeo_johnson_transformer = PowerTransformer(method="yeo-johnson")
        else:
            yeo_johnson_transformer = transformer

        yeo_johnson_features = [
            'number_of_instances',
            'number_of_classes',
            'ratio_of_instances_to_features',
            'ratio_of_classes_to_features',
            'ratio_of_instances_to_classes',
            'average_mutual_information',
            'maximum_mutual_information',
            'noise_to_signal_ratio_of_features',
            'proportion_of_numeric_features',
            'minimum_mutual_information',
            'equivalent_number_of_features',
            'learning_rate',
            'number_of_hidden_layers',
            'min_number_of_neurons',
            'total_number_of_neurons',
            'avg_learning_slope',
            'std_learning_slope',
            'avg_validation_gap',
            'std_validation_gap',
            'avg_first_overfit_epoch',
            'std_first_overfit_epoch',
            'avg_first_stay_epoch',
            'std_first_stay_epoch',
            'avg_percentage_weights_near_zero',
            'std_percentage_weights_near_zero',
            'avg_avg_weights',
            'std_avg_weights',
            'depth_to_width_ratio',
            'batch_size_to_dataset_ratio'
        ]

        if features is not None:
            features = features.copy()

            yeo_johnson_cols_to_transform = [col for col in yeo_johnson_features if col in features.columns]
            if yeo_johnson_cols_to_transform:
                if transformer != None:
                    features[yeo_johnson_cols_to_transform] = yeo_johnson_transformer.transform(features[yeo_johnson_cols_to_transform])
                else:
                    features[yeo_johnson_cols_to_transform] = yeo_johnson_transformer.fit_transform(features[yeo_johnson_cols_to_transform])

            # if 'equivalent_number_of_features' in features.columns:
            #     x = features['equivalent_number_of_features']
            #     features['equivalent_number_of_features'] = np.log(x - 1 + 1e-6)
            return features, yeo_johnson_transformer
        elif training_features is not None and testing_features is not None:
            training_features = training_features.copy()
            testing_features = testing_features.copy()

            yeo_johnson_cols_to_transform = [col for col in yeo_johnson_features if col in training_features.columns]

            if yeo_johnson_cols_to_transform:
                if transformer != None:
                    training_features[yeo_johnson_cols_to_transform] = yeo_johnson_transformer.transform(training_features[yeo_johnson_cols_to_transform])
                else:
                    training_features[yeo_johnson_cols_to_transform] = yeo_johnson_transformer.fit_transform(training_features[yeo_johnson_cols_to_transform])
                testing_features[yeo_johnson_cols_to_transform] = yeo_johnson_transformer.transform(testing_features[yeo_johnson_cols_to_transform])

            # if 'equivalent_number_of_features' in training_features.columns:
            #     training_features['equivalent_number_of_features'] = np.log(training_features['equivalent_number_of_features'] - 1 + 1e-6)
            #     testing_features['equivalent_number_of_features'] = np.log(testing_features['equivalent_number_of_features'] - 1 + 1e-6)

            return training_features, testing_features, yeo_johnson_transformer
        else:
            assert False, "Either dataset or both training_set and testing_set must be provided."

def apply_normalization(features=None, training_features=None, testing_features=None, ignore_columns = [], scaler = None):
    if scaler is None:
        z_scoring_scaler = StandardScaler()
    else:
        z_scoring_scaler = scaler
    if features is not None:
        features = features.copy()
        for column in features.columns:
            if column not in ignore_columns:
                if scaler is not None:
                    features[column] = z_scoring_scaler.transform(features[column].values.reshape(-1, 1)).flatten()
                else:
                    features[column] = z_scoring_scaler.fit_transform(features[column].values.reshape(-1, 1)).flatten()
        return features, z_scoring_scaler
    elif training_features is not None and testing_features is not None:
        if set(training_features.columns) != set(testing_features.columns):
            raise ValueError("Training and testing features must have the same columns")
        training_features = training_features.copy()
        testing_features = testing_features.copy()
        for column in training_features.columns:
            if column not in ignore_columns:
                if scaler is not None:
                    training_features[column] = z_scoring_scaler.transform(training_features[column].values.reshape(-1, 1)).flatten()
                else:
                    training_features[column] = z_scoring_scaler.fit_transform(training_features[column].values.reshape(-1, 1)).flatten()
                testing_features[column] = z_scoring_scaler.transform(testing_features[column].values.reshape(-1, 1)).flatten()
        return training_features, testing_features, z_scoring_scaler
    else:
        raise ValueError("Either dataset or both training_set and testing_set must be provided.")

def rank_techniques_mann_whitney(targets):
    assert targets.shape[1] >= 2, "Need at least two techniques to compare."

    # apply hypothesis test
    def row_wise_pval_matrix(row):
        return pd.DataFrame({
            col1: [apply_pValue(row[col1], row[col2]) for col2 in targets.columns]
            for col1 in targets.columns
        }, index=targets.columns)

    print("Calculating the pvals of each cell.")
    pvals_matrices = targets.apply(row_wise_pval_matrix, axis=1)
    # calculate mean
    print("Calculating the mean of each cell.")
    means_dataset = targets.applymap(calculate_mean)

    # Apply custom ranking with equivalence (pval >= 0.5 means not significantly different)
    ranked_rows = []
    print("Ranking the columns for each row.")
    for idx, row in means_dataset.iterrows():
        pval_matrix = pvals_matrices.loc[idx]
        means = row.to_dict()
        remaining = set()
        ranks = {}
        for col in row.index:
            if math.isinf(row[col]):
                ranks[col] = -1
            else:
                remaining.add(col)

        rank = 1
        while remaining:
            group = []
            ref = min(remaining, key=lambda x: means[x])
            for other in list(remaining):
                if pval_matrix.loc[ref, other]:
                    group.append(other)
            for technique in group:
                ranks[technique] = rank
                remaining.remove(technique)
            rank += 1
        ranked_rows.append(ranks)

    return pd.DataFrame(ranked_rows, index=means_dataset.index)

def rank_techniques_friedman(targets):
    assert targets.shape[1] >= 2, "Need at least two techniques to compare."

    techniques = list(targets.columns)
    ranked_rows = pd.DataFrame(columns=techniques)

    alpha = 0.1

    for idx, row in targets.iterrows():
        ranked_row = {technique: [] for technique in techniques}
        values_by_technique = {
            technique: np.asarray(cell_parse(row[technique]), dtype=float)
            for technique in techniques
        }

        run_count = len(next(iter(values_by_technique.values())))
        if any(len(values) != run_count for values in values_by_technique.values()):
            raise ValueError(f"All technique columns must contain the same number of values for row {idx}.")

        # 1) Rank techniques across the 10 runs (higher F1 = better rank)
        for counter in range(run_count):
            run_values = {technique: values[counter] for technique, values in values_by_technique.items()}
            ranked_techniques = sorted(run_values, key=run_values.get, reverse=True)
            for rank, technique in enumerate(ranked_techniques, start=1):
                ranked_row[technique].append(rank)

        # 2) Friedman test across all samples
        statistic, p_value = friedmanchisquare(
            *[ranked_row[technique] for technique in techniques]
        )

        for technique in techniques:
            if p_value > 0.1:
                ranked_row[technique] = 1
            else:
                ranked_row[technique] = round(np.mean(ranked_row[technique]))

        if p_value <= 0.1:
            # 3) Nemenyi pairwise comparisons
            if studentized_range is not None:
                q_critical = studentized_range.ppf(1 - alpha, len(techniques), np.inf) / np.sqrt(2)
                critical_difference = q_critical * np.sqrt(
                    len(techniques) * (len(techniques) - 1) / (6 * run_count)
                )
            else:
                critical_difference = 0.0

            changed = True
            while changed:
                changed = False
                for technique1, technique2 in combinations(techniques, 2):
                    diff = abs(ranked_row[technique1] - ranked_row[technique2])
                    significant = diff > critical_difference if critical_difference > 0 else False
                    if not significant:
                        if ranked_row[technique1] < ranked_row[technique2]:
                            ranked_row[technique2] = ranked_row[technique1]
                            changed = True
                        elif ranked_row[technique2] < ranked_row[technique1]:
                            ranked_row[technique1] = ranked_row[technique2]
                            changed = True

        valid_ranks = sorted({rank for rank in ranked_row.values() if rank >= 1})
        rank_map = {rank: index for index, rank in enumerate(valid_ranks, start=1)}
        ranked_row = {
            technique: rank_map[rank] if rank >= 1 else rank
            for technique, rank in ranked_row.items()
        }

        ranked_rows.loc[idx] = ranked_row

    return pd.DataFrame(ranked_rows, index=targets.index)


def cell_parse(cell):
    if isinstance(cell, list):
        values = cell
    else:
        cell1_fixed = cell.replace('inf', '"inf"')
        values = ast.literal_eval(cell1_fixed)
    values = [float('inf') if value == 'inf' else value for value in values]
    return values

def apply_pValue(cell1, cell2):
    if cell1 == cell2:
        return True
    values1 = cell_parse(cell1)
    values2 = cell_parse(cell2)

    # Handle cases with inf values
    if any(math.isinf(v) for v in values1) and any(math.isinf(v) for v in values2):
        return True
    elif any(math.isinf(v) for v in values1) or any(math.isinf(v) for v in values2):
        return False

    stat, p_value = ttest_ind(values1, values2, equal_var=False)
    return p_value >= 0.1

def calculate_mean(cell):
    values = cell_parse(cell)
    return sum(values) / len(values)

def show_dataset_loader_menu(allow_full_dataset = False, return_both_sets = False):
    if allow_full_dataset:
        was_processed= input("Has the dataset been processed before? (y/n): ").lower() == "y"
        set_types = ["Full dataset", "Training dataset", "Validation dataset"]
        set_type = show_menu("What type of dataset do you want to calculate stats for? ", set_types)
        if was_processed:
            if set_type == set_types[0]:
                return load_meta_features_csv()
            else:
                return load_meta_features_csv(set_type.split(" ")[0].strip().lower())
        else:
            if set_type == set_types[0]:
                return prepare_meta_feature_dataset_for_states()
            else:
                training_set, validation_set = prepare_meta_feature_sets()
                if set_type == set_types[1]:
                    return training_set
                else:
                    return validation_set
    elif return_both_sets:
            if input("Do you have training and validation sets? (y/n): ").lower() == "y":
                return load_meta_features_csv("training"), load_meta_features_csv("validation")
            else:
                return prepare_meta_feature_sets()
    else:
        if input("Do you have training sets? (y/n): ").lower() == "y":
            return load_meta_features_csv("training")
        else:
            training_set, _ = prepare_meta_feature_sets()
            return training_set