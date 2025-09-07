import warnings as wr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Utils.metaFeatureDatasetHandler import targetColumns, spiltDatasetAndTargets

wr.filterwarnings('ignore')

def calculateDatasetStats(fullDataset):
    dataset, targets = spiltDatasetAndTargets(fullDataset)

    pd.set_option("display.max_columns", None)
    print(f"Shape: {dataset.shape}")
    print("")

    print(f"Columns:")
    print(dataset.info())
    print("")

    print(f"Description:")
    infRemoved = dataset.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    description = infRemoved.describe()
    for column in description.columns.tolist():
        print(infRemoved.describe()[column])
    print("")
    for column in targetColumns:
        print(targets.describe()[column])
    print("")

    #Make Technique rankings summary
    worst_counts = targets.idxmax(axis=1).value_counts().reindex(targets.columns, fill_value=0)

    summary = pd.DataFrame({
        'average_rank': targets.mean(),
        'best_count': (targets == 1).sum(),
        'best_count_percent': ((targets == 1).sum()/targets.shape[0]*100),
        'worst_count': worst_counts,
        'worst_count_percent': (worst_counts/targets.shape[0]*100)
    })

    print("Technique rankings summary:")
    print(summary)

    #Make Technique count bar chart
    print("Making technique count bar chart")

    num_cols = 3
    num_rows = int(np.ceil(len(targetColumns) / num_cols))

    plt.figure(figsize=(12, num_rows * 3))
    plt.title('Bar chart of technique rankings')

    for idx, column in enumerate(targetColumns, start=1):
        ranking_counts = targets[column].value_counts()
        # Ensure the index is numeric, then sort
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
    plt.title('Density plot of meta features')

    for idx, feature in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, num_cols, idx)
        clean_data = dataset[feature].replace([np.inf, -np.inf], np.nan).dropna()
        sns.histplot(clean_data, kde=True)
        plt.title(f"{feature}\nSkewness: {round(dataset[feature].skew(), 2)}")

    plt.tight_layout()
    plt.show()

    #Show outlier
    print("Making box plot")
    sns.set_style("darkgrid")

    numerical_columns_base = fullDataset.select_dtypes(include=["int64", "float64", "int8"]).columns
    numerical_columns = [col for col in numerical_columns_base if col not in targetColumns]

    for technique in targetColumns:
        plt.figure(figsize=(15, num_rows * 3))
        plt.suptitle('Box charts of meta features vs ' + technique)

        for idx, feature in enumerate(numerical_columns, 1):
            plt.subplot(num_rows, num_cols, idx)
            sns.boxplot(y=feature, x=technique, data=fullDataset)  # Ensure 'technique' is string/categorical
            plt.title(f"{feature}")

        plt.tight_layout()
        plt.show()

    #Bivariate Analysis
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