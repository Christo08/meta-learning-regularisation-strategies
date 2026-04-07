import os
import warnings as wr
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.Utils.constants import TARGET_COLUMNS, STATS_OPTIONS
from src.Utils.fileHandler import load_results_csv, save_data_frame
from src.Utils.menus import show_menu
from src.Utils.metaFeatureDatasetHandler import spilt_dataset_and_targets

wr.filterwarnings('ignore')


def create_feature_stats(features, output_path):
    print(f"Making the feature summary")
    inf_removed = features.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    stats_df = inf_removed.describe().T
    stats_df = stats_df.round(3)
    stats_df = stats_df.reset_index().rename(columns={'index': 'column name'})
    stats_df = stats_df[['column name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    save_data_frame(stats_df, f"{output_path}\\features_stats.csv")
    print(f"The feature summary was save to {output_path}\\features_stats.csv")


def create_technique_rankings_stats(targets, output_path):
    print(f"Making the technique summary")
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

    save_data_frame(summary, f"{output_path}\\rankings_stats.csv")
    print(f"The technique summary was save to {output_path}\\rankings_stats.csv")


def create_technique_stack_bar_chart(full_dataset, output_path):
    print("Making technique count bar chart")
    selected_columns = ['dataset_name'] + TARGET_COLUMNS
    subset = full_dataset[selected_columns]
    rankings_per_dataset = (
        subset.groupby('dataset_name')[TARGET_COLUMNS]
        .apply(lambda x: (x == 1).sum())
        .reset_index()
    )
    save_data_frame(rankings_per_dataset, f"{output_path}\\rankings_per_dataset.csv")

    fig, ax = plt.subplots(figsize=(10, 15))

    rankings_per_dataset.plot(
        x="dataset_name",
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title('Stack bar chart of technique rankings')
    ax.tick_params(axis='x', labelrotation=90)

    fig.tight_layout()
    fig.savefig(f"{output_path}\\rankings_bar_chart.png", dpi=300)
    plt.show()
    print(f"The technique count bar chart was save to {output_path}\\rankings_bar_chart.png")


def create_feature_density_plots(features, output_path):
    print("Making features density plot")
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
    plt.savefig(f"{output_path}\\features_density_plots.png")
    plt.show()
    print(f"The features density plot was save to {output_path}\\features_density_plots.png")


def create_box_plots(full_dataset, output_path):
    print("Making box plot")
    sns.set_style("darkgrid")

    numerical_columns_base = full_dataset.select_dtypes(include=["int64", "float64", "int8"]).columns
    numerical_columns = [col for col in numerical_columns_base if col not in TARGET_COLUMNS]

    records = []

    for tech in TARGET_COLUMNS:
        if tech not in full_dataset.columns:
            continue

        df_t = full_dataset[full_dataset[tech] == 1]

        for feature in numerical_columns:
            for val in df_t[feature].dropna():
                records.append({
                    "technique": tech,
                    "feature": feature,
                    "value": val,
                })

    plot_df = pd.DataFrame(records)

    features = plot_df["feature"].unique()

    num_cols = 3
    num_rows = int(np.ceil(len(features) / num_cols))
    height_per_row = 5
    plt.figure(figsize=(12, num_rows * height_per_row))
    plt.title('Box plots of features vs techniques')
    for idx, feature in enumerate(features, 1):
        plt.subplot(num_rows, num_cols, idx)
        df_f = plot_df[plot_df["feature"] == feature]

        sns.boxplot(
            data=df_f,
            x="technique",
            y="value",
        )

        plt.title(f"{feature}")
        plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(f"{output_path}\\features_box_plots.png")
    plt.show()
    print(f"The features box plots was save to {output_path}\\features_box_plots.png")


def create_pair_plot(features, targets, output_path):
    if not targets.empty:
        features = pd.concat([features, targets], axis=1)

    print("Making pair plot")
    sns.set_palette("Pastel1")

    plot = sns.pairplot(features)

    plot.figure.suptitle('Pair Plot of Features vs Techniques')
    plot.figure.savefig(f"{output_path}\\pair_plot.png")
    plot.figure.show()
    print(f"The pair plot was save to {output_path}\\pair_plot.png")


def create_heatmap(features, targets, output_path):
    if not targets.empty:
        features = pd.concat([features, targets], axis=1)
        if "dataset_name" in features.columns:
            features.drop("dataset_name", axis=1, inplace=True)

        print("Making correlation heatmap")
        plt.figure(figsize=(25, 20))

        sns.heatmap(features.corr(), annot=True, fmt='.2f', cmap='Greys', linewidths=2)

        plt.title('Correlation Heatmap')
        plt.savefig(f"{output_path}\\correlation_heatmap.png")
        plt.show()
        print(f"The correlation heatmap was save to {output_path}\\correlation_heatmap.png")


def calculate_dataset_stats(full_dataset):
    features, targets = spilt_dataset_and_targets(full_dataset)
    has_target = not targets.empty
    output_path = input("Enter the path of the Output stats folder: ")

    prompt ="Select the stats to calculate by entering the numbers "+ "" if has_target else "(Options 1,2 and 3 are not available)"
    processes_option = show_menu(prompt, STATS_OPTIONS)
    selected_processes = []
    if processes_option == STATS_OPTIONS[0]:
        selected_processes =  STATS_OPTIONS[1:-2]
    elif processes_option == STATS_OPTIONS[len(STATS_OPTIONS) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        for select_dataset_index in select_dataset_indexes:
            selected_processes.append(processes_option[int(select_dataset_index) - 1])
    elif processes_option == STATS_OPTIONS[len(STATS_OPTIONS) - 1]:
        return
    else:
        selected_processes = [processes_option]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_path}\\{timestamp}"
    os.makedirs(output_path, exist_ok=True)

    pd.set_option("display.max_columns", None)

    if STATS_OPTIONS[5] in selected_processes:
        create_feature_stats(features, output_path)

    if STATS_OPTIONS[6] in selected_processes and has_target:
        create_technique_rankings_stats(targets, output_path)

    #Make Technique count bar chart
    if STATS_OPTIONS[1] in selected_processes and has_target:
        create_technique_stack_bar_chart(full_dataset, output_path)

    #Density plot of columns
    if STATS_OPTIONS[4] in selected_processes:
        create_feature_density_plots(features, output_path)

    #Show outlier
    if STATS_OPTIONS[2] in selected_processes and has_target:
        create_box_plots(full_dataset, output_path)

    #Bivariate Analysis
    if STATS_OPTIONS[7] in selected_processes:
        create_pair_plot(features, targets, output_path)

    #Correlation Heatmap
    if STATS_OPTIONS[3] in selected_processes:
        create_heatmap(features, targets, output_path)

def calculate_meta_learners_stats():
    meta_learners_results = load_results_csv()
    output_path = input("Enter the path of the Output stats folder: ")

    print("Making the f1 training box plots:")
    show_meta_learners_box_plots(meta_learners_results, 'training f1', output_path)

    print("Making the confusion matrix:")
    create_confusion_matrix(meta_learners_results, output_path)

    print("Making the f1 testing bar chart")
    create_meta_learners_bar_charts(meta_learners_results, 'testing f1', output_path)

def create_confusion_matrix(dataset, output_path):
    required_cols = [
        "model type",
        "technique",
        "testing true positives",
        "testing true negatives",
        "testing false positives",
        "testing false negatives",
    ]
    missing = [c for c in required_cols if c not in dataset.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df =dataset.copy()

    # Convert to numeric
    for c in required_cols[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Now build one figure per technique; each figure has one subplot per model type
    techniques = list(df["technique"].dropna().unique())
    model_types = list(df["model type"].dropna().unique())

    for technique in techniques:
        technique_df = df[df["technique"] == technique]

        # Make a grid for up to 5 models (you said 5)
        number_of_models = len(model_types)
        number_cols = 3
        number_rows = int(np.ceil(number_of_models / number_cols))
        fig, axes = plt.subplots(number_rows, number_cols, figsize=(14, number_rows * 5))
        axes = np.array(axes).reshape(-1)

        for idx, model_type in enumerate(model_types):
            ax = axes[idx]
            module_df = technique_df[technique_df["model type"] == model_type]

            # Get the mean of the folds
            tp = float(module_df["testing true positives"].iloc[0])
            tn = float(module_df["testing true negatives"].iloc[0])
            fp = float(module_df["testing false positives"].iloc[0])
            fn = float(module_df["testing false negatives"].iloc[0])

            confusion_matrix = np.array([[tn, fp], [fn, tp]])

            sns.heatmap(confusion_matrix, annot=True, fmt=".0f",  cmap="Blues", ax=ax)
            ax.set_title(f"{model_type}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        # delete unused axes
        for counter in range(len(model_types), len(axes)):
            fig.delaxes(axes[counter])

        fig.suptitle(f"Confusion Matrices — Technique: {technique}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        file_path = f"{output_path}\\confusion_matrices_{str(technique).replace(' ', '_')}.png"
        fig.savefig(file_path, dpi=300)
        plt.show()

        print(f"Saved confusion matrices for technique '{technique}' to {file_path}")

def create_meta_learners_bar_charts(meta_learners_results, metric_column_name, output_path):
    df = meta_learners_results.copy()

    # Ensure numeric y values
    df[metric_column_name] = pd.to_numeric(df[metric_column_name], errors="coerce")
    df = df.dropna(subset=[metric_column_name, "technique", "model type"])

    sns.set_style("darkgrid")

    techniques = list(df["technique"].dropna().unique())
    if len(techniques) == 0:
        raise ValueError("No techniques found after filtering; cannot plot bar charts.")

    # Make a grid of subplots (one subplot per technique)
    ncols = 3
    nrows = int(np.ceil(len(techniques) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 5))
    axes = np.array(axes).reshape(-1)  # safe even if nrows/ncols == 1

    for i, technique in enumerate(techniques):
        ax = axes[i]
        tdf = df[df["technique"] == technique].copy()

        # Optional: stable ordering
        tdf = tdf.sort_values("model type", ascending=False)

        sns.barplot(
            data=tdf,
            x="model type",
            y=metric_column_name,
            errorbar=None,
            color="steelblue",
            ax=ax,
        )

        ax.set_title(f"Technique: {technique}")
        ax.set_xlabel("Model type")
        ax.set_ylabel(metric_column_name)
        ax.tick_params(axis="x", rotation=90)

    # Remove/hide unused subplots
    for j in range(len(techniques), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{metric_column_name} by model type (all techniques)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    safe_metric = metric_column_name.replace(" ", "_")
    file_path = f"{output_path}\\{safe_metric}_bar_chart.png"
    fig.savefig(file_path, dpi=300)
    plt.show()


def show_meta_learners_box_plots(meta_learners_results, metric_column_name, output_path):
    sns.set_style("darkgrid")

    techniques = list(meta_learners_results['technique'].unique())
    number_cols = 3
    number_rows = int(np.ceil(len(techniques) / number_cols))

    fig, axes = plt.subplots(number_rows, number_cols, figsize=(14, number_rows * 5))
    axes = axes.flatten()

    for idx, technique in enumerate(techniques):
        group = meta_learners_results[meta_learners_results['technique'] == technique]

        if metric_column_name not in group.columns:
            continue
        group_exp = group[['model type', metric_column_name]].reset_index(drop=True)
        group_exp = explode_accuracies(group_exp, metric_column_name)
        group_exp[metric_column_name] = group_exp[metric_column_name].astype(float)

        stats = group_exp.groupby('model type')[metric_column_name].agg(['mean', 'min', 'max', 'std'])
        print(f"\nModel type: {technique} - {metric_column_name.capitalize()} statistics:")
        print(stats)

        # Plotting boxplot for testing accuracies
        group_plot = group[['model type', metric_column_name]].reset_index(drop=True)
        group_plot = explode_accuracies(group_plot, metric_column_name)
        group_plot[metric_column_name] = group_plot[metric_column_name].astype(float)

        ax = axes[idx]
        sns.boxplot(x='model type', y=metric_column_name, data=group_plot, ax=ax)
        ax.set_title(f'{technique}')
        ax.set_ylabel(metric_column_name)
        ax.set_xlabel('Model type')
        ax.tick_params(axis='x', rotation=90)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Boxplot of {metric_column_name} by Model Type for Each Technique', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    file_path = f"{output_path}\\{metric_column_name.replace(" ","_")}_box_plot.png"
    plt.savefig(file_path, dpi=300)
    plt.show()
    print(f'Saved {metric_column_name}\'s box plot to {file_path}')

def explode_accuracies(df, accuracy_col):
    df = df.copy()
    df[accuracy_col] = df[accuracy_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return df.explode(accuracy_col)

def get_best_instances_for_techniques(full_dataset):
    best_instances = {}
    for technique in TARGET_COLUMNS:
        other_techniques = [t for t in TARGET_COLUMNS if t != technique]
        best_instances[technique] = full_dataset[full_dataset[technique] == 1].drop(columns=other_techniques)
    return best_instances

def calculated_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()

    assert true_negatives + false_positives + false_negatives + true_positives == len(y_true)

    return true_positives, true_negatives, false_positives, false_negatives