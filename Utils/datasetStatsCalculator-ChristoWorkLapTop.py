import ast
import math
import warnings as wr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from networkx.algorithms.community.quality import intra_community_edges
from scipy.stats import friedmanchisquare, levene, ttest_1samp
import seaborn as sns
from scipy.stats import ttest_ind

wr.filterwarnings('ignore')

from Utils.fileHandler import loadMetaFeaturesCSV

def loadDataset():
    dataset = loadMetaFeaturesCSV()

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
        "best_validation_technique"
    ]

    dataset.drop(columns=columns_to_drop, errors="ignore", inplace=True)

    dataset.rename(columns={"best_testing_technique": "technique"}, inplace=True)
    dataset['technique'] = dataset['technique'].astype('category')
    dataset['technique'] = dataset['technique'].cat.codes

    return dataset


def calculateDatasetStats():
    dataset = loadDataset()
    targetColumns = ["normal_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                      "layer_normalisation_testing_loss", "SMOTE_testing_loss", "prune_testing_loss",
                      "weight_decay_testing_loss", "weight_normalisation_testing_loss", "weight_perturbation_testing_loss"]
    targets = dataset[targetColumns]
    dataset = dataset.drop(targetColumns, axis=1)
    targets = rankTechniques(targets)

    pd.set_option("display.max_columns", None)
    print(f"Shape: {dataset.shape}")
    print("")

    print(f"Columns:")
    print(dataset.info())
    print("")

    print(f"Description:")
    description = dataset.describe()
    for column in description.columns.tolist():
        print(dataset.describe()[column])
    print("")
    for column in targetColumns:
        print(targets.describe()[column])
    print("")

    #Make Technique count bar chart
    print("Making technique count bar chart")

    num_cols = 3
    num_rows = int(np.ceil(len(targetColumns) / num_cols))

    plt.figure(figsize=(12, num_rows * 3))

    for idx, column in enumerate(targetColumns, start=1):
        ranking_counts = targets[column].value_counts()
        # Ensure index is numeric, then sort
        ranking_counts.index = ranking_counts.index.astype(int)
        ranking_counts = ranking_counts.sort_index()

        plt.subplot(num_rows, num_cols, idx)
        plt.bar(ranking_counts.index.astype(str), ranking_counts.values)
        plt.title('Count ranking of ' + column)
        plt.xlabel(column)
        plt.xticks(ticks=range(1, 10))  # explicitly set 1 to 9

    plt.tight_layout()
    plt.show()

    #Density plot of columns
    print("Making density plot")
    sns.set_style("darkgrid")

    numerical_columns = dataset.select_dtypes(include=["int64", "float64", "int8"]).columns

    num_cols = 3
    num_rows = int(np.ceil(len(numerical_columns) / num_cols))

    plt.figure(figsize=(12, num_rows * 3))

    for idx, feature in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, num_cols, idx)
        sns.histplot(dataset[feature], kde=True)
        plt.title(f"{feature}\nSkewness: {round(dataset[feature].skew(), 2)}")

    plt.tight_layout()
    plt.show()

    #Show outlier
    print("Making box plot")
    sns.set_style("darkgrid")

    numerical_columns = dataset.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(15, num_rows * 3))
    for idx, feature in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, num_cols, idx)
        sns.boxplot(y=feature, x="technique", data=dataset)  # Ensure 'technique' is string/categorical
        plt.title(f"{feature}")

    plt.tight_layout()
    plt.show()

    #Bivariate Analysis
    dataset = dataset.drop("technique",axis=1)
    dataset = pd.concat([dataset, targets], axis=1)
    print("Making pair plot")
    sns.set_palette("Pastel1")

    plt.figure(figsize=(10, 6))

    sns.pairplot(dataset)

    plt.suptitle('Pair Plot for DataFrame')
    plt.show()

    #Correlation Heatmap
    print("Making correlation heatmap")
    plt.figure(figsize=(25, 20))

    sns.heatmap(dataset.corr(), annot=True, fmt='.2f', cmap='Greys', linewidths=2)

    plt.title('Correlation Heatmap')
    plt.show()

def rankTechniques(dataset: pd.DataFrame):
    assert dataset.shape[1] >= 2, "Need at least two techniques to compare."

    # apply hypothesis test
    pvalsDataset = []
    columnNames = []
    for refColumn in dataset.columns:
        refColumn = dataset.columns[0]
        pvalsColumn = dataset.apply(lambda row: [applyTtest(row[refColumn], row[col]) for col in dataset.columns], axis=1,
                            result_type='expand')
        pvalsColumn.columns = dataset.columns
        pvalsDataset.append(pvalsColumn)
        columnNames.append(refColumn)

    # calculate mean
    meansDataset = dataset.applymap(calculate_mean)

    rankedData = meansDataset.rank(axis=1, method='dense', ascending=True)

    worst_counts = rankedData.idxmax(axis=1).value_counts().reindex(meansDataset.columns, fill_value=0)

    summary = pd.DataFrame({
        'average_rank': rankedData.mean(),
        'best_count': (rankedData == 1).sum(),
        'worst_count': worst_counts
    })

    print(summary)
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
    values1 = cellParse(cell1)
    values2 = cellParse(cell2)

    # Handle cases with inf values
    if any(math.isinf(v) for v in values1) and any(math.isinf(v) for v in values2):
        return False
    elif any(math.isinf(v) for v in values1) or any(math.isinf(v) for v in values2):
        return True

    stat, p_value = ttest_ind(values1, values2, equal_var=False)
    return p_value >= 0.5

def calculate_mean(cell):
    values = cellParse(cell)
    return sum(values) / len(values)