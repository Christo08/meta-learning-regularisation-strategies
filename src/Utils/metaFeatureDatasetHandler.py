import ast
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, zscore
from sklearn.model_selection import train_test_split

from src.Utils.constants import TARGET_COLUMNS
from src.Utils.fileHandler import load_meta_features_csv, save_data_frame


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

    set_output_path = input("Enter the path to folder to which the set should be saved: ")
    stats_output_path = input("Enter the path of the output stats folder: ")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Train datasets:")
    train_datasets_name = train["dataset_name"].tolist()
    print(train_datasets_name)
    training_set = dataset[dataset["dataset_name"].isin(train_datasets_name)]
    save_data_frame(training_set, f"{set_output_path}\\training_set_{timestamp}.csv")
    training_set_stats_df = training_set.describe().T
    training_set_stats_df = training_set_stats_df.reset_index().rename(columns={'index': 'column name'})
    training_set_stats_df = training_set_stats_df[['column name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    save_data_frame(training_set_stats_df, f"{stats_output_path}\\training_set_stats_{timestamp}.csv")

    print("Test datasets:")
    test_datasets_name = test["dataset_name"].tolist()
    print(test_datasets_name)
    testing_set = dataset[dataset["dataset_name"].isin(test_datasets_name)]
    save_data_frame(testing_set, f"{set_output_path}\\testing_set_{timestamp}.csv")
    testing_set_stats_df = testing_set.describe().T
    testing_set_stats_df = testing_set_stats_df.reset_index().rename(columns={'index': 'column name'})
    testing_set_stats_df = testing_set_stats_df[['column name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    save_data_frame(testing_set_stats_df, f"{stats_output_path}\\testing_set_stats_{timestamp}.csv")

def load_meta_feature_dataset(need_subsets_info = False, type ="", should_cover_to_binary = False, should_ask_for_apply_z_scoring = True, should_ask_rank_techniques = True):
    should_rank_techniques = input("Is the dataset raw? (y/n): ").lower() == "y" if should_ask_rank_techniques else False
    dataset = load_meta_features_csv(type)
    dataset = clean_dataset(dataset)
    for target_column in TARGET_COLUMNS:
        if target_column not in dataset.columns:
            raise ValueError(f"Missing target column: {target_column}")
    if should_rank_techniques:
        targets = dataset[TARGET_COLUMNS]
        dataset = dataset.drop(TARGET_COLUMNS, axis=1)

        targets = rank_techniques(targets)

        dataset = pd.concat([dataset, targets], axis=1)

        output_path = input("Enter the path of the Output dataset folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"regularisation_meta_learning_{timestamp}.csv"
        file_path = output_path + "\\" + file_name
        save_data_frame(dataset, file_path)
    if should_cover_to_binary:
        dataset = dataset.drop(columns=["dataset_name", "SMOTE"], errors="ignore", inplace=False)
        dataset = apply_z_scoring(dataset, should_ask_for_apply_z_scoring)
        if not need_subsets_info:
            dataset.drop(columns=["dataset_name"], errors="ignore", inplace=True)
        for column in TARGET_COLUMNS:
            if column in dataset.columns:
                dataset[column] = dataset[column].apply(lambda x: 1 if x == 1 else 0)
    return dataset

def clean_dataset(dataset):
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
        "weight_perturbation_testing_accuracies", "seed", "file_name", "subset_type"
    ]

    dataset.drop(columns=columns_to_drop, errors="ignore", inplace=True)

    suffix = "_testing_loss"
    rename_map = {
        col: col[:-len(suffix)]
        for col in dataset.columns
        if isinstance(col, str) and col.endswith(suffix)
    }
    dataset.rename(columns=rename_map, inplace=True)

    for column in dataset.columns:
        if not(column in TARGET_COLUMNS) and column != "dataset_name":
            dataset[column] = dataset[column].values.astype(np.float64)

    return dataset

def apply_z_scoring(dataset, should_ask_for_apply_z_scoring):
    should_apply_z_scoring = input("Apply Z scoring? (y/n): ").lower() == "y" if should_ask_for_apply_z_scoring else False
    if should_apply_z_scoring:
        max_float = np.finfo(np.float32).max
        for column in dataset.columns:
            if not(column in TARGET_COLUMNS) and column != "dataset_name":
                column_data = dataset[column].values.astype(np.float64)
                finite_mask = np.isfinite(column_data)

                z_column = np.empty_like(column_data)
                z_column[:] = column_data
                z_column[finite_mask] = zscore(column_data[finite_mask])

                z_column = np.where(z_column == np.inf, max_float, z_column)

                dataset[column] = z_column
    return dataset


def rank_techniques(dataset):
    assert dataset.shape[1] >= 2, "Need at least two techniques to compare."

    # apply hypothesis test
    def row_wise_pval_matrix(row):
        return pd.DataFrame({
            col1: [apply_ttest(row[col1], row[col2]) for col2 in dataset.columns]
            for col1 in dataset.columns
        }, index=dataset.columns)

    print("Calculating the pvals of each cell.")
    pvals_matrices = dataset.apply(row_wise_pval_matrix, axis=1)
    # calculate mean
    print("Calculating the mean of each cell.")
    means_dataset = dataset.applymap(calculate_mean)

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
