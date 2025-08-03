import ast
import math

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, zscore
from sklearn.model_selection import train_test_split
from datetime import datetime

from Utils.fileHandler import loadMetaFeaturesCSV, saveMetaFeaturesDataset

targetColumns = ["normal_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                 "layer_normalisation_testing_loss", "SMOTE_testing_loss", "prune_testing_loss",
                 "weight_decay_testing_loss", "weight_normalisation_testing_loss", "weight_perturbation_testing_loss"]

def spiltDatasetAndTargets(dataset):
    targets = dataset[targetColumns]
    dataset = dataset.drop(targetColumns, axis=1)
    return dataset, targets

def createTestingSet(dataset, targetColumn, seed):
    subset, testingSet = train_test_split(
        dataset, test_size=0.2, random_state=seed
    )
    testingSetX, testingSetY = spiltDatasetAndTargets(testingSet)
    testingSetY = testingSetY[targetColumn]
    subsetX, subsetY = spiltDatasetAndTargets(subset)
    subsetY = subsetY[targetColumn]
    return (subsetX, subsetY), (testingSetX, testingSetY)

def loadMetaFeatureDataset():
    shouldRankTechniques = input("Is the dataset raw? (y/n): ").lower() == "y"
    dataset = loadMetaFeaturesCSV()
    if shouldRankTechniques:
        dataset = cleanDataset(dataset)

        targets = dataset[targetColumns]
        dataset = dataset.drop(targetColumns, axis=1)

        targets = rankTechniques(targets)

        dataset = pd.concat([dataset, targets], axis=1)

        outputPath = input("Enter the path of the Output dataset folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fileName = f"processed_meta_feature_{timestamp}.csv"
        filePath = outputPath + "\\" + fileName
        saveMetaFeaturesDataset(dataset, filePath)
    return dataset

def cleanDataset(dataset):
    columns_to_drop = [
        "batch_size", "dropout_layers", "learning_rate", "momentum", "number_of_epochs", "number_of_hidden_layers",
        "number_of_neurons_in_layers", "prune_amount", "prune_epoch_interval", "weight_decay",
        "weight_perturbation_amount", "weight_perturbation_interval", "dataset_name", "seed", "normal_training_loss",
        "normal_validation_loss", "batch_normalisation_training_loss", "batch_normalisation_validation_loss",
        "dropout_training_loss", "dropout_validation_loss", "layer_normalisation_training_loss",
        "layer_normalisation_validation_loss", "SMOTE_training_loss", "SMOTE_validation_loss", "prune_training_loss",
        "prune_validation_loss", "weight_decay_training_loss", "weight_decay_validation_loss",
        "weight_normalisation_training_loss", "weight_normalisation_validation_loss",
        "weight_perturbation_training_loss", "weight_perturbation_validation_loss", "best_training_technique",
        "best_validation_technique", "best_testing_technique"
    ]

    dataset.drop(columns=columns_to_drop, errors="ignore", inplace=True)
    maxFloat = np.finfo(np.float32).max

    for column in dataset.columns:
        if not(column in targetColumns):
            columnData = dataset[column].values.astype(np.float64)

            finiteMask = np.isfinite(columnData)

            zColumn = np.empty_like(columnData)
            zColumn[:] = columnData
            zColumn[finiteMask] = zscore(columnData[finiteMask])

            zColumn = np.where(zColumn == np.inf, maxFloat, zColumn)

            dataset[column] = zColumn

    return dataset

def rankTechniques(dataset):
    assert dataset.shape[1] >= 2, "Need at least two techniques to compare."

    # apply hypothesis test
    def rowWisePvalMatrix(row):
        return pd.DataFrame({
            col1: [applyTtest(row[col1], row[col2]) for col2 in dataset.columns]
            for col1 in dataset.columns
        }, index=dataset.columns)

    print("Calculating the pvals of each cell.")
    pvalsMatrices = dataset.apply(rowWisePvalMatrix, axis=1)
    # calculate mean
    print("Calculating the mean of each cell.")
    meansDataset = dataset.applymap(calculate_mean)

    # Apply custom ranking with equivalence (pval >= 0.5 means not significantly different)
    ranked_rows = []
    print("Ranking the columns for each row.")
    for idx, row in meansDataset.iterrows():
        pvalMatrix = pvalsMatrices.loc[idx]
        means = row.to_dict()
        remaining = set(means.keys())
        ranks = {}
        rank = 1
        while remaining:
            group = []
            ref = min(remaining, key=lambda x: means[x])
            for other in list(remaining):
                if pvalMatrix.loc[ref, other]:
                    group.append(other)
            for technique in group:
                ranks[technique] = rank
                remaining.remove(technique)
            rank += 1
        ranked_rows.append(ranks)

    rankedData = pd.DataFrame(ranked_rows, index=meansDataset.index)

    return rankedData

def cellParse(cell):
    if isinstance(cell, list):
        values = cell
    else:
        cell1_fixed = cell.replace('inf', '"inf"')
        values = ast.literal_eval(cell1_fixed)
    values = [float('inf') if value == 'inf' else value for value in values]
    return values

def applyTtest(cell1, cell2):
    if cell1 == cell2:
        return True
    values1 = cellParse(cell1)
    values2 = cellParse(cell2)

    # Handle cases with inf values
    if any(math.isinf(v) for v in values1) and any(math.isinf(v) for v in values2):
        return True
    elif any(math.isinf(v) for v in values1) or any(math.isinf(v) for v in values2):
        return False

    stat, p_value = ttest_ind(values1, values2, equal_var=False)
    return p_value >= 0.5

def calculate_mean(cell):
    values = cellParse(cell)
    return sum(values) / len(values)
