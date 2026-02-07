import warnings as wr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Utils.fileHandler import load_results_csv
from Utils.metaFeatureDatasetHandler import target_columns, spilt_dataset_and_targets

wr.filterwarnings('ignore')

def calculate_dataset_stats(full_dataset):
    dataset, targets = spilt_dataset_and_targets(full_dataset)
    has_target = not targets.empty

    pd.set_option("display.max_columns", None)
    print(f"Shape: {dataset.shape}")
    print("")

    print(f"Columns:")
    print(dataset.info())
    print("")

    print(f"Description:")
    inf_removed = dataset.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    description = inf_removed.describe()
    for column in description.columns.tolist():
        print(inf_removed.describe()[column])
    print("")

    if has_target:
        for column in target_columns:
            print(targets[column].describe())

        print("")

    #Make Technique rankings summary
    if has_target:
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
    if has_target:
        print("Making technique count bar chart")

        num_cols = 3
        num_rows = int(np.ceil(len(target_columns) / num_cols))

        plt.figure(figsize=(12, num_rows * 3))
        plt.title('Bar chart of technique rankings')

        for idx, column in enumerate(target_columns, start=1):
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
    if has_target:
        print("Making box plot")
        sns.set_style("darkgrid")

        numerical_columns_base = full_dataset.select_dtypes(include=["int64", "float64", "int8"]).columns
        numerical_columns = [col for col in numerical_columns_base if col not in target_columns]

        for technique in target_columns:
            plt.figure(figsize=(15, num_rows * 3))
            plt.suptitle('Box charts of meta features vs ' + technique)

            for idx, feature in enumerate(numerical_columns, 1):
                plt.subplot(num_rows, num_cols, idx)
                sns.boxplot(y=feature, x=technique, data=full_dataset)  # Ensure 'technique' is string/categorical
                plt.title(f"{feature}")

            plt.tight_layout()
            plt.show()

    #Bivariate Analysis
    if has_target:
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

def calculate_meta_learners_stats():
    meta_learners_results = load_results_csv()

    print("Meta Learners Results Info:")
    print(meta_learners_results.info())

    print("Meta Learners Results Correlation:")
    sns.set_style("darkgrid")

    techniques = list(meta_learners_results['technique'].unique())
    number_cols = 3
    number_rows = int(np.ceil(len(techniques) / number_cols))

    fig, axes = plt.subplots(number_rows, number_cols, figsize=(14, number_rows * 5))
    axes = axes.flatten()

    for idx, technique in enumerate(techniques):
        group = meta_learners_results[meta_learners_results['technique'] == technique]

        for accuracy_col in ['testing accuracies', 'training accuracies']:
            if accuracy_col not in group.columns:
                continue
            group_exp = group[['model type', accuracy_col]].reset_index(drop=True)
            group_exp = explode_accuracies(group_exp, accuracy_col)
            group_exp[accuracy_col] = group_exp[accuracy_col].astype(float)

            stats = group_exp.groupby('model type')[accuracy_col].agg(['mean', 'min', 'max', 'std'])
            print(f"\nModel type: {technique} - {accuracy_col.capitalize()} statistics:")
            print(stats)

        # Plotting boxplot for testing accuracies
        if 'testing accuracies' in group.columns:
            group_plot = group[['model type', 'testing accuracies']].reset_index(drop=True)
            group_plot = explode_accuracies(group_plot, 'testing accuracies')
            group_plot['testing accuracies'] = group_plot['testing accuracies'].astype(float)

            ax = axes[idx]
            sns.boxplot(x='model type', y='testing accuracies', data=group_plot, ax=ax)
            ax.set_title(f'{technique}')
            ax.set_ylabel('Testing Accuracies')
            ax.set_xlabel('Model type')
            ax.tick_params(axis='x', rotation=90)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Boxplot of Testing Accuracies by Model Type for Each Technique', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def explode_accuracies(df, accuracy_col):
    df = df.copy()
    df[accuracy_col] = df[accuracy_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return df.explode(accuracy_col)