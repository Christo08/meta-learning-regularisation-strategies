import ast
import math
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, zscore

from Utils.fileHandler import load_meta_features_csv, save_data_frame

target_columns = [
    "baseline_testing_loss",
    "batch_normalisation_testing_loss",
    "dropout_testing_loss",
    "layer_normalisation_testing_loss",
    "prune_testing_loss",
    "weight_normalisation_testing_loss"
]

def spilt_dataset_and_targets(dataset):
    missing = False
    for targetColumn in target_columns:
        if targetColumn not in dataset.columns:
            missing = True
            break
    if missing:
        targets = pd.DataFrame()
        return dataset, targets
    else:
        targets = dataset[target_columns]
        dataset = dataset.drop(target_columns, axis=1)
        return dataset, targets

def load_meta_feature_dataset(need_subsets_info = False, type ="", should_cover_to_binary = False):
    should_rank_techniques = input("Is the dataset raw? (y/n): ").lower() == "y"
    dataset = load_meta_features_csv(type)
    missing = False
    if not need_subsets_info:
        columns_to_drop = ["dataset_name", "seed", "file_name", "subset_type"]
        dataset.drop(columns=columns_to_drop, errors="ignore", inplace=True)
    for target_column in target_columns:
        if target_column not in dataset.columns:
            missing = True
            break
    if should_rank_techniques and not(missing):
        dataset = clean_dataset(dataset)

        targets = dataset[target_columns]
        dataset = dataset.drop(target_columns, axis=1)

        targets = rank_techniques(targets)

        dataset = pd.concat([dataset, targets], axis=1)

        output_path = input("Enter the path of the Output dataset folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"processed_meta_feature_{timestamp}.csv"
        file_path = output_path + "\\" + file_name
        save_data_frame(dataset, file_path)
    if should_cover_to_binary:
        for column in target_columns:
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
        "weight_perturbation_testing_accuracies"
    ]
    should_apply_z_scoring = input("Apply Z scoring? (y/n): ").lower() == "y"

    dataset.drop(columns=columns_to_drop, errors="ignore", inplace=True)
    max_float = np.finfo(np.float32).max

    for column in dataset.columns:
        if not(column in target_columns):
            column_data = dataset[column].values.astype(np.float64)
            if should_apply_z_scoring:
                finite_mask = np.isfinite(column_data)

                z_column = np.empty_like(column_data)
                z_column[:] = column_data
                z_column[finite_mask] = zscore(column_data[finite_mask])

                z_column = np.where(z_column == np.inf, max_float, z_column)

                dataset[column] = z_column
            else:
                dataset[column] = column_data


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
