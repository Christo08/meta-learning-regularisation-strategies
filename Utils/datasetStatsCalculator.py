import warnings as wr
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from Utils.fileHandler import load_results_csv, save_data_frame
from Utils.constants import META_LEANER_TARGET_COLUMNS, STATS_OPTIONS
from Utils.metaFeatureDatasetHandler import spilt_dataset_and_targets
from Utils.menus import show_menu

wr.filterwarnings('ignore')
process_options = ["Basic", "Cluster analysis", "Back"]
dimensional_reduction_options = ["TSNE", "UMAP", "Back"]

def calculate_stats(full_dataset):
    process = show_menu("Select stats type: ", process_options)
    if process == process_options[0]:
        calculate_dataset_stats(full_dataset)
    elif process == process_options[1]:
        perform_cluster_analysis(full_dataset)
    else:
        return

def perform_cluster_analysis(full_dataset):
    option = show_menu("Select dimensional reduction technique: ", dimensional_reduction_options)
    if option in dimensional_reduction_options[2]:
        return
    best_instances_for_techniques = get_best_instances_for_techniques(full_dataset)
    for key, instances in best_instances_for_techniques.items():
        X_pca = PCA(n_components=10).fit_transform(instances)
        if option == dimensional_reduction_options[0]:
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate=200,
                init='pca',
                random_state=42
            )

            X_embedded = tsne.fit_transform(X_pca)
        else:
            umap_model = umap.UMAP(
                n_neighbors=15,
                min_dist=0.05,
                n_components=2,
                metric='euclidean',
                random_state=42
            )

            X_umap = umap_model.fit_transform(X_pca)

def calculate_dataset_stats(full_dataset):
    dataset_without_datasets_names = full_dataset.drop(columns=["dataset_name"], errors="ignore")
    features, targets = spilt_dataset_and_targets(dataset_without_datasets_names)
    has_target = not targets.empty
    output_path = input("Enter the path of the Output stats folder: ")
    prompt ="Select the stats to calculate by entering the numbers "+ "" if has_target else "(Options 1,2 and 3 are not available)"
    processes_option = show_menu(prompt, STATS_OPTIONS)
    selected_processes = []
    if processes_option == STATS_OPTIONS[0]:
        selected_processes =  STATS_OPTIONS[1:-2]
    elif processes_option == STATS_OPTIONS[len(processes_option) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        for select_dataset_index in select_dataset_indexes:
            selected_processes.append(processes_option[int(select_dataset_index) - 1])
    elif processes_option == STATS_OPTIONS[len(processes_option) - 1]:
        return
    else:
        selected_processes = [processes_option]

    pd.set_option("display.max_columns", None)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if STATS_OPTIONS[5] in selected_processes:
        print(f"Features Description:")
        inf_removed = features.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
        stats_df = inf_removed.describe().T
        stats_df = stats_df.round(3)
        stats_df = stats_df.reset_index().rename(columns={'index': 'column name'})
        stats_df = stats_df[['column name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        save_data_frame(stats_df, f"{output_path}/features_stats_{timestamp}.csv")
        print(stats_df)
        print("")

    if STATS_OPTIONS[6] in selected_processes and has_target:
        print(f"Technique Description:")

        for column in META_LEANER_TARGET_COLUMNS:
            print(targets[column].describe())
        print("")

        #Make Technique rankings summary
        worst_counts = targets.idxmax(axis=1).value_counts().reindex(targets.columns, fill_value=0)
        can_not_be_applied = (targets == -1).sum()

        summary = pd.DataFrame({
            'average_rank': round(targets.mean(),3),
            'best_count': (targets == 1).sum(),
            'best_count_percent': round(((targets == 1).sum()/(targets.shape[0]-can_not_be_applied)*100),3),
            'worst_count': worst_counts,
            'worst_count_percent': round((worst_counts/(targets.shape[0]-can_not_be_applied)*100),3),
            'can_not_be_used_count': can_not_be_applied,
            'can_not_be_used_percent': round((can_not_be_applied/targets.shape[0]*100),3)
        })
        summary = summary.reset_index().rename(columns={'index': 'technique'})

        print("Technique rankings summary:")
        print(summary)
    #Make Technique count bar chart
    if STATS_OPTIONS[1] in selected_processes and has_target:
        print("Making technique count bar chart")
        save_data_frame(summary, f"{output_path}/rankings_stats_{timestamp}.csv")
        selected_columns = ['dataset_name'] + META_LEANER_TARGET_COLUMNS
        subset = full_dataset[selected_columns]
        rankings_per_dataset = subset.groupby('dataset_name')[META_LEANER_TARGET_COLUMNS].apply(lambda x: (x == 1).sum()).reset_index()
        save_data_frame(rankings_per_dataset, f"{output_path}/rankings_per_dataset_{timestamp}.csv")


        plt.figure(figsize=(50,50))
        rankings_per_dataset.plot(x="dataset_name", kind='bar', stacked=True)
        plt.title('Stack bar chart of technique rankings')
        plt.savefig(f"{output_path}/rankings_bar_chart_{timestamp}.png")
        plt.show()

    #Density plot of columns
    if STATS_OPTIONS[4] in selected_processes:
        print("Making density plot")
        sns.set_style("darkgrid")

        numerical_columns = features.select_dtypes(include=["int64", "float64", "int8"]).columns

        num_cols = 3
        num_rows = int(np.ceil(len(numerical_columns) / num_cols))

        plt.figure(figsize=(12, num_rows * 3))
        plt.title('Density plot of meta features')

        for idx, feature in enumerate(numerical_columns, 1):
            plt.subplot(num_rows, num_cols, idx)
            clean_data = features[feature].replace([np.inf, -np.inf], np.nan).dropna()
            sns.histplot(clean_data, kde=True)
            plt.title(f"{feature}\nSkewness: {round(features[feature].skew(), 2)}")

        plt.tight_layout()
        plt.savefig(f"{output_path}/features_density_plots_{timestamp}.png")
        plt.show()

    #Show outlier
    if STATS_OPTIONS[2] in selected_processes and has_target:
        print("Making box plot")
        sns.set_style("darkgrid")

        numerical_columns_base = full_dataset.select_dtypes(include=["int64", "float64", "int8"]).columns
        numerical_columns = [col for col in numerical_columns_base if col not in META_LEANER_TARGET_COLUMNS]

        for technique in META_LEANER_TARGET_COLUMNS:
            plt.figure(figsize=(15, num_rows * 3))
            plt.suptitle('Box charts of meta features vs ' + technique)

            for idx, feature in enumerate(numerical_columns, 1):
                plt.subplot(num_rows, num_cols, idx)
                sns.boxplot(y=feature, x=technique, data=full_dataset)  # Ensure 'technique' is string/categorical
                plt.title(f"{feature}")

            plt.tight_layout()
            plt.savefig(f"{output_path}/features_vs_{technique}_box_plots_{timestamp}.png")
            plt.show()

    #Bivariate Analysis
    if STATS_OPTIONS[7] in selected_processes:
        if has_target:
            features = pd.concat([features, targets], axis=1)

        print("Making pair plot")
        sns.set_palette("Pastel1")

        plt.figure(figsize=(10, 6))

        sns.pairplot(features)

        plt.suptitle('Pair Plot of Meta Features vs Techniques')
        plt.show()

    #Correlation Heatmap
    if STATS_OPTIONS[3] in selected_processes:
        print("Making correlation heatmap")
        plt.figure(figsize=(25, 20))

        sns.heatmap(features.corr(), annot=True, fmt='.2f', cmap='Greys', linewidths=2)

        plt.title('Correlation Heatmap')
        plt.savefig(f"{output_path}/correlation_heatmap_{timestamp}.png")
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

def get_best_instances_for_techniques(full_dataset):
    best_instances = {}
    for technique in META_LEANER_TARGET_COLUMNS:
        other_techniques = [t for t in META_LEANER_TARGET_COLUMNS if t != technique]
        best_instances[technique] = full_dataset[full_dataset[technique] == 1].drop(columns=other_techniques)
    return best_instances