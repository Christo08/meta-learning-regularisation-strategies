import ast
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.Utils.constants import TARGET_COLUMNS
from src.Utils.fileHandler import load_meta_features_csv, save_data_frame, get_latest_settings


def spilt_dataset_and_targets(dataset):
    missing = False
    for target_column in TARGET_COLUMNS:
        if target_column not in dataset.columns:
            missing = True
            break
    if missing:
        targets = pd.DataFrame()
        return dataset, targets
    else:
        targets = dataset[TARGET_COLUMNS]
        dataset = dataset.drop(TARGET_COLUMNS, axis=1)
        return dataset, targets

def split_dataset(dataset):
    targets = [col for col in TARGET_COLUMNS if col != "SMOTE"]
    selected_columns = ['dataset_name'] + targets
    subset = dataset[selected_columns]
    rankings_per_dataset = subset.groupby('dataset_name')[targets].apply(lambda x: (x == 1).sum()).reset_index()

    rankings_per_dataset["mean_loss"] = rankings_per_dataset[targets].mean(axis=1)
    rankings_per_dataset["bin"] = pd.qcut(rankings_per_dataset["mean_loss"], q=4, labels=False)

    seed = random.randint(1, 10000)
    print("Seed:", seed)
    train, test = train_test_split(rankings_per_dataset,
                                   test_size=0.25,
                                   stratify=rankings_per_dataset["bin"],
                                   random_state = seed)

    train_datasets_name = train["dataset_name"].tolist()
    test_datasets_name = test["dataset_name"].tolist()

    print("Train datasets:")
    print(train_datasets_name)

    print("Test datasets:")
    print(test_datasets_name)
    training_set = dataset[dataset["dataset_name"].isin(train_datasets_name)]
    testing_set = dataset[dataset["dataset_name"].isin(test_datasets_name)]

    return training_set, testing_set

def add_hyperparameters(dataset):
    for index, row in dataset.iterrows():
        settings = get_latest_settings(row["dataset_name"])
        if not settings:
            raise ValueError(f"Missing setting: {row}")
        dataset.loc[index,"batch_size"] = settings["batch_size"]
        dataset.loc[index,"learning_rate"] = settings["learning_rate"]
        dataset.loc[index,"number_of_epochs"] = settings["number_of_epochs"]
        dataset.loc[index,"number_of_hidden_layers"] = settings["number_of_hidden_layers"]
        number_of_neurons = settings["number_of_neurons_in_layers"][:settings["number_of_hidden_layers"]]
        dataset.loc[index,"avg_number_of_neurons"] = np.average(number_of_neurons)
        dataset.loc[index,"min_number_of_neurons"] = np.min(number_of_neurons)
        dataset.loc[index,"max_number_of_neurons"] = np.max(number_of_neurons)
    return dataset

def prepare_meta_feature_dataset_for_states():
    dataset = load_meta_features_csv("")

    options = ""
    should_add_hyperparameters = input("Do you want to add hyperparameters (y/n): ").lower() == "y"
    if should_add_hyperparameters:
        options = "hyperparameters_"
        dataset = add_hyperparameters(dataset)
    
    dataset = clean_dataset(dataset, False)

    should_create_bins = input("Do you want to create bins? (y/n): ").lower() == "y"
    if should_create_bins:
        options = options+"bins_"
        bin_config = fit_binning(dataset)
        dataset = apply_binning(dataset, bin_config)

    targets = dataset[TARGET_COLUMNS]
    features = dataset.drop(TARGET_COLUMNS, axis=1)

    should_apply_yeo_johnson = input("Do you want to apply Yeo-Johnson transformation? (y/n): ").lower() == "y"
    if should_apply_yeo_johnson:
        options = options+"yeo_johnson_"
        features = apply_yeo_johnson(features = features)

    should_normalise= input("Do you want to normalise the dataset? (y/n): ").lower() == "y"
    if should_normalise:
        options = options+"normalised_"
        features = apply_normalization(features=features)

    targets = rank_techniques(targets)
    should_cover_to_binary = input("Do you want to convert the ranks to binary (1 for best technique, 0 for others)? (y/n): ").lower() == "y"
    if should_cover_to_binary:
        options = options+"binary_"
        for column in TARGET_COLUMNS:
            if column in targets.columns:
                targets[column] = targets[column].apply(lambda x: 1 if x == 1 else 0)

    dataset = pd.concat([features, targets], axis=1)

    should_save_dataset = input("Do you want to save the prepared dataset? (y/n): ").lower() == "y"
    if should_save_dataset:
        output_path = input("Enter the path of the output dataset folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"regularisation_meta_learning_{options}{timestamp}.csv"
        file_path = output_path + "\\" + file_name
        save_data_frame(dataset, file_path)

    return dataset

def prepare_meta_feature_sets():
    was_processed = input("Has the dataset be processed before, note normalise and bins should not have been applied? (y/n): ").lower() == "y"
    dataset = load_meta_features_csv("")
    if not was_processed:
        dataset = clean_dataset(dataset, False)
        targets = dataset[TARGET_COLUMNS]
        features = dataset.drop(TARGET_COLUMNS, axis=1)
        targets = rank_techniques(targets)
        should_cover_to_binary = input("Do you want to convert the ranks to binary (1 for best technique, 0 for others)? (y/n): ").lower() == "y"
        if should_cover_to_binary:
            for column in TARGET_COLUMNS:
                if column in targets.columns:
                    targets[column] = targets[column].apply(lambda x: 1 if x == 1 else 0)
        dataset = pd.concat([features, targets], axis=1)

    options = ""
    should_add_hyperparameters = input("Do you want to add hyperparameters (y/n): ").lower() == "y"
    if should_add_hyperparameters:
        options = "hyperparameters_"
        dataset = add_hyperparameters(dataset)
    training_set, testing_set = split_dataset(dataset)

    should_create_bins = input("Do you want to create bins? (y/n): ").lower() == "y"
    if should_create_bins:
        options = options+"bins_"
        bin_config = fit_binning(training_set)
        training_set = apply_binning(training_set, bin_config)
        testing_set = apply_binning(testing_set, bin_config)

    training_targets = training_set[TARGET_COLUMNS]
    training_features = training_set.drop(TARGET_COLUMNS, axis=1)

    testing_targets = testing_set[TARGET_COLUMNS]
    testing_features = testing_set.drop(TARGET_COLUMNS, axis=1)

    training_features, testing_features = apply_normalization(training_features = training_features, testing_features = testing_features, are_there_bins= should_create_bins)

    training_set = pd.concat([training_features, training_targets], axis=1)
    testing_set = pd.concat([testing_features, testing_targets], axis=1)

    should_save_dataset = input("Do you want to save the prepared dataset? (y/n): ").lower() == "y"
    if should_save_dataset:
        output_path = input("Enter the path of the output dataset folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"regularisation_meta_learning_testing_set_{options}{timestamp}.csv"
        file_path = output_path + "\\" + file_name
        save_data_frame(testing_set, file_path)
        file_name = f"regularisation_meta_learning_training_set_{options}{timestamp}.csv"
        file_path = output_path + "\\" + file_name
        save_data_frame(training_set, file_path)

    return training_set, testing_set

def clean_dataset(dataset, should_drop_dataset_name = True):
    columns_to_drop = [
        "baseline_training_loss", "baseline_validation_loss", "batch_normalisation_training_loss",
        "batch_normalisation_validation_loss", "dropout_training_loss", "dropout_validation_loss",
        "layer_normalisation_training_loss", "layer_normalisation_validation_loss", "SMOTE_training_loss",
        "SMOTE_validation_loss", "prune_training_loss", "prune_validation_loss", "weight_decay_training_loss",
        "weight_decay_validation_loss", "weight_normalisation_training_loss", "weight_normalisation_validation_loss",
        "weight_perturbation_training_loss", "weight_perturbation_validation_loss", "best_training_technique",
        "best_validation_technique", "best_testing_technique","baseline_training_accuracies","baseline_testing_accuracies",
        "batch_normalisation_training_accuracies",
        "batch_normalisation_testing_accuracies","dropout_training_accuracies","dropout_testing_accuracies",
        "layer_normalisation_training_accuracies","layer_normalisation_testing_accuracies","SMOTE_training_accuracies",
        "SMOTE_testing_accuracies","prune_training_accuracies","prune_testing_accuracies",
        "weight_decay_training_accuracies","weight_decay_testing_accuracies","weight_normalisation_training_accuracies",
        "weight_normalisation_testing_accuracies","weight_perturbation_training_accuracies",
        "weight_perturbation_testing_accuracies", "seed","file_name", "subset_type", "SMOTE_testing_loss"
    ]
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

def append_hyperparameters(dataset):
    for index, row in dataset.iterrows():
        setting = get_latest_settings(row["dataset_name"])
        if not setting:
            raise ValueError(f"Missing setting: {row}")
        dataset.loc[index, "batch_size"] = setting["batch_size"]
        dataset.loc[index, "learning_rate"] = setting["learning_rate"]
        dataset.loc[index, "number_of_epochs"] = setting["number_of_epochs"]
        dataset.loc[index, "number_of_hidden_layers"] = setting["number_of_hidden_layers"]
        number_of_neurons = setting["number_of_neurons_in_layers"][:setting["number_of_hidden_layers"]]
        dataset.loc[index, "avg_number_of_neurons"] = np.average(number_of_neurons)
        dataset.loc[index, "min_number_of_neurons"] = np.min(number_of_neurons)
        dataset.loc[index, "max_number_of_neurons"] = np.max(number_of_neurons)
    return dataset

def apply_yeo_johnson(features=None, training_features=None, testing_features=None):
        transformer = PowerTransformer(method="yeo-johnson")

        if features is not None:
            transformer.fit(features)
            transformed = transformer.transform(features)
            features = pd.DataFrame(transformed, index=features.index, columns=features.columns)
            return features

        elif training_features is not None and testing_features is not None:
            transformer.fit(training_features)
            training_transformed = transformer.transform(training_features)
            training_transformed = pd.DataFrame(training_transformed, index=training_features.index, columns=training_features.columns)
            testing_transformed = transformer.transform(testing_features)
            testing_transformed = pd.DataFrame(testing_transformed, index=testing_features.index, columns=testing_features.columns)

            return training_transformed, testing_transformed
        else:
            assert False, "Either dataset or both training_set and testing_set must be provided."

def apply_normalization(features=None, training_features=None, testing_features=None, are_there_bins = False):
    bins_columns = [
        'number_of_instances',
        'ratio_of_instances_to_features',
        'ratio_of_instances_to_classes',
        'noise_to_signal_ratio_of_features',
        'average_mutual_information',
        'maximum_mutual_information',
        'equivalent_number_of_features',
        'minimum_mutual_information',
        'ratio_of_classes_to_features'
    ] if are_there_bins else []
    if features is not None:
        features = features.copy()
        for column in features.columns:
            if column not in bins_columns and column != "dataset_name":
                scaler = StandardScaler()
                features[column] = scaler.fit_transform(features[column].values.reshape(-1, 1)).flatten()
        return features
    elif training_features is not None and testing_features is not None:
        if set(training_features.columns) != set(testing_features.columns):
            raise ValueError("Training and testing features must have the same columns")
        training_features = training_features.copy()
        testing_features = testing_features.copy()
        for column in training_features.columns:
            if column not in bins_columns and column != "dataset_name":
                scaler = StandardScaler()
                training_features[column] = scaler.fit_transform(training_features[column].values.reshape(-1, 1)).flatten()
                testing_features[column] = scaler.transform(testing_features[column].values.reshape(-1, 1)).flatten()
        return training_features, testing_features
    else:
        raise ValueError("Either dataset or both training_set and testing_set must be provided.")

def rank_techniques(targets):
    assert targets.shape[1] >= 2, "Need at least two techniques to compare."

    # apply hypothesis test
    def row_wise_pval_matrix(row):
        return pd.DataFrame({
            col1: [apply_ttest(row[col1], row[col2]) for col2 in targets.columns]
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

def cell_parse(cell):
    if isinstance(cell, list):
        values = cell
    else:
        cell1_fixed = cell.replace('inf', '"inf"')
        values = ast.literal_eval(cell1_fixed)
    values = [float('inf') if value == 'inf' else value for value in values]
    return values

def apply_ttest(cell1, cell2):
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
    return p_value >= 0.5

def calculate_mean(cell):
    values = cell_parse(cell)
    return sum(values) / len(values)

def fit_binning(dataset: pd.DataFrame):
    bin_config = {}

    log_features = [
        'number_of_instances',
        'ratio_of_instances_to_features',
        'ratio_of_instances_to_classes',
        'noise_to_signal_ratio_of_features'
    ]

    for col in log_features:
        x = dataset[col].copy()

        x = x[x > 0]

        min_val = x.min()
        max_val = x.max()

        edges = np.logspace(np.log10(min_val), np.log10(max_val), 6)
        bin_config[col] = edges

    quantile_features = [
        'average_mutual_information',
        'maximum_mutual_information',
        'equivalent_number_of_features'
    ]

    for col in quantile_features:
        edges = np.quantile(dataset[col], q=[0, 0.25, 0.5, 0.75, 1.0])
        bin_config[col] = np.unique(edges)

    col = 'minimum_mutual_information'
    non_zero = dataset[col][dataset[col] > 0]

    if len(non_zero) > 0:
        edges_non_zero = np.quantile(non_zero, q=[0, 0.33, 0.66, 1.0])
        bin_config[col] = {
            'zero_bin': True,
            'edges': np.unique(edges_non_zero)
        }

    bin_config['ratio_of_classes_to_features'] = np.array([0, 0.2, 0.5, 1, 2, np.inf])

    return bin_config

def apply_binning(dataset: pd.DataFrame, bin_config: dict):
    binned_df = dataset.copy()

    for col, config in bin_config.items():

        if isinstance(config, dict) and config.get('zero_bin', False):
            binned = np.zeros(len(dataset))

            non_zero_mask = dataset[col] > 0
            binned[non_zero_mask] = pd.cut(
                dataset.loc[non_zero_mask, col],
                bins=config['edges'],
                labels=False,
                include_lowest=True
            ).fillna(0) + 1  # shift because 0 is reserved

            binned_df[col] = binned.astype(int)
            continue

        # --- normal binning ---
        binned_df[col] = pd.cut(
            dataset[col],
            bins=config,
            labels=False,
            include_lowest=True
        ).fillna(0)

    return binned_df