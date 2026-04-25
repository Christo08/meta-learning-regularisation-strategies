import os
import warnings as wr
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr, stats
from scipy.stats import ttest_rel
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
    stats_df = stats_df.round(4)
    stats_df = stats_df.reset_index().rename(columns={'index': 'column name'})
    stats_df['skewness'] = inf_removed.skew(numeric_only=True).round(4).values
    stats_df = stats_df[['column name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness']]
    if output_path is not None:
        save_data_frame(stats_df, f"{output_path}\\features_stats.csv")
        print(f"The feature summary was save to {output_path}\\features_stats.csv")


def create_technique_rankings_stats(targets, output_path):
    print(f"Making the technique summary")
    #Make Technique rankings summary
    worst_counts = targets.idxmax(axis=1).value_counts().reindex(targets.columns, fill_value=0)
    can_not_be_applied = (targets == -1).sum()

    summary = pd.DataFrame({
        'average_rank': round(targets.mean(),2),
        'best_count': (targets == 1).sum(),
        'best_count_percent': round(((targets == 1).sum()/(targets.shape[0]-can_not_be_applied)*100),2),
        'worst_count': worst_counts,
        'worst_count_percent': round((worst_counts/(targets.shape[0]-can_not_be_applied)*100),2)
    })
    summary = summary.reset_index().rename(columns={'index': 'technique'})
    if output_path is not None:
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

    number_of_rows = full_dataset.shape[0]
    percent_of_ones = (subset[TARGET_COLUMNS] == 1).sum() / number_of_rows *100

    fig, ax = plt.subplots(figsize=(10, 15))
    percent_of_ones.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title('Percentage of Instances in Which Each Technique Was the Best Performer')
    ax.set_ylabel('Percentage (%)')
    ax.tick_params(axis='x', labelrotation=90)

    fig.tight_layout()
    if output_path is not None:
        save_data_frame(rankings_per_dataset, f"{output_path}\\rankings_per_dataset.csv")
        fig.savefig(f"{output_path}\\rankings_bar_chart.png", dpi=300)
        print(f"The technique count bar chart was save to {output_path}\\rankings_bar_chart.png")
    plt.show()


def create_feature_density_plots(features, output_path):
    print("Making features density plot")
    sns.set_style("darkgrid")

    numerical_columns = features.select_dtypes(include=["int64", "float64", "int8"]).columns

    num_cols = 3
    num_rows = int(np.ceil(len(numerical_columns) / num_cols))

    plt.figure(figsize=(12, num_rows * 3))
    plt.suptitle('Density plot of meta features', y=1)  # Changed from plt.title() to plt.suptitle() with y parameter

    for idx, feature in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, num_cols, idx)
        clean_data = features[feature].replace([np.inf, -np.inf], np.nan).dropna()
        sns.histplot(clean_data, kde=True)
        plt.title(f"{feature}\nSkewness: {round(features[feature].skew(), 2)}")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if output_path is not None:# Add rect parameter to leave room for suptitle
        plt.savefig(f"{output_path}\\features_density_plots.png", bbox_inches='tight')  # Add bbox_inches to include the title
        print(f"The features density plot was save to {output_path}\\features_density_plots.png")
    plt.show()


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
    plt.figure(figsize=(18, num_rows * 6))  # Changed from num_rows * 5 to num_rows * 3
    plt.suptitle('Box plots of features vs techniques', y=1)
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

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if output_path is not None:
        plt.savefig(f"{output_path}\\features_box_plots.png", bbox_inches='tight')
        print(f"The features box plots was save to {output_path}\\features_box_plots.png")  # Added bbox_inches
    plt.show()


def create_pair_plot(features, targets, output_path):
    if not targets.empty:
        features = pd.concat([features, targets], axis=1)

    print("Making pair plot")
    sns.set_palette("Pastel1")

    plot = sns.pairplot(features)

    plot.figure.suptitle('Pair Plot of Features vs Techniques')
    if output_path is not None:
        plot.figure.savefig(f"{output_path}\\pair_plot.png")
        print(f"The pair plot was save to {output_path}\\pair_plot.png")
    plot.figure.show()


def create_heatmap(features, targets, output_path):
    if not targets.empty:
        if "dataset_name" in features.columns:
            features.drop("dataset_name", axis=1, inplace=True)

        correlations_matrix = features.corr()

        for target_column in targets.columns:
            for feature_column in features.columns:
                correlation,_ = pointbiserialr(targets[target_column], features[feature_column])
                correlations_matrix.loc[feature_column, target_column] = correlation
                correlations_matrix.loc[target_column, feature_column] = correlation

        for target_column in targets.columns:
            for target_column_two in targets.columns:
                correlation,_ = pointbiserialr(targets[target_column], targets[target_column_two])
                correlations_matrix.loc[target_column_two, target_column] = correlation

        print("Making correlation heatmap")
        plt.figure(figsize=(25, 20))

        sns.heatmap(correlations_matrix, annot=True, fmt='.2f', cmap='Greys', linewidths=2)

        plt.title('Correlation Heatmap')
        if output_path is not None:
            plt.savefig(f"{output_path}\\correlation_heatmap.png")
            print(f"The correlation heatmap was save to {output_path}\\correlation_heatmap.png")
        plt.show()


def calculate_dataset_stats(full_dataset):
    features, targets = spilt_dataset_and_targets(full_dataset)
    has_target = not targets.empty
    should_save = input("Do you want to save the stats to a file? (y/n): ").lower() == 'y'
    if should_save:
        output_path = input("Enter the path of the output stats folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_path}\\{timestamp}"
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = None

    prompt ="Select the stats to calculate by entering the numbers "+ "" if has_target else "(Options 1,2 and 3 are not available)"
    processes_option = show_menu(prompt, STATS_OPTIONS)
    selected_processes = []
    if processes_option == STATS_OPTIONS[0]:
        selected_processes =  STATS_OPTIONS[1:-2]
    elif processes_option == STATS_OPTIONS[len(STATS_OPTIONS) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        for select_dataset_index in select_dataset_indexes:
            selected_processes.append(STATS_OPTIONS[int(select_dataset_index) - 1])
    elif processes_option == STATS_OPTIONS[len(STATS_OPTIONS) - 1]:
        return
    else:
        selected_processes = [processes_option]

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
    meta_learners_results.drop(columns=["model path"], inplace=True, errors='ignore')
    should_save = input("Do you want to save the stats to a file? (y/n): ").lower() == 'y'
    if should_save:
        output_path = input("Enter the path of the output stats folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_path}\\{timestamp}"
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = None

    print("Making the f1 training box plots:")
    show_meta_learners_box_plots(meta_learners_results, 'training f1', output_path)

    print("Making the f1 testing bar chart")
    create_meta_learners_bar_charts(meta_learners_results, 'testing f1', output_path)

    print("Making the training confusion matrix:")
    create_confusion_matrix(meta_learners_results, output_path, "training")

    print("Making the testing confusion matrix:")
    create_confusion_matrix(meta_learners_results, output_path, "testing")

def calculate_meta_learners_performance():
    meta_learners_results = load_results_csv()
    should_save = input("Do you want to save the stats to a file? (y/n): ").lower() == 'y'
    if should_save:
        output_path = input("Enter the path of the output stats folder: ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_path}\\{timestamp}"
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = None

    meta_learners_results.drop(columns=["seed","best_technique"], inplace=True, errors='ignore')
    meta_learners_results = meta_learners_results.drop(
        columns=[col for col in meta_learners_results.columns if 'training' in col]
    )

    create_meta_learner_comparison_boxplots(meta_learners_results, output_path, 'f1 scores')

    summaries_results(meta_learners_results, output_path)
    normalise_result(meta_learners_results, output_path)

    create_f1_comparison_heatmap(meta_learners_results, save_path=output_path)

def normalise_result(meta_learners_results, output_path):
    normalised_df = meta_learners_results.copy()
    f1_scores_columns = [col for col in normalised_df.columns if 'f1_scores' in col and 'testing' in col]

    normalised_df["baseline_testing_f1_scores_mean"] = normalised_df["baseline_testing_f1_scores"].apply(
        lambda x: np.mean(eval(x)) if isinstance(x, str) else np.mean(x)
    )

    for col in f1_scores_columns:
        normalised_df[f"{col}_normalized"] = normalised_df.apply(
            lambda row: [
                f1_val / row["baseline_testing_f1_scores_mean"]
                for f1_val in (eval(row[col]) if isinstance(row[col], str) else row[col])
            ],
            axis=1
        )

    normalized_columns = [f"{col}_normalized" for col in f1_scores_columns]
    columns_to_keep = ['dataset_name'] + normalized_columns
    normalised_df = normalised_df[columns_to_keep]

    techniques = [col.replace('_testing_f1_scores_normalized', '') for col in normalized_columns]
    num_datasets = len(normalised_df)

    num_cols = 3
    num_rows = int(np.ceil(num_datasets / num_cols))

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 5))
    axes = np.array(axes).reshape(-1)

    for idx, (row_idx, row) in enumerate(normalised_df.iterrows()):
        ax = axes[idx]
        plot_data = []

        for norm_col, technique in zip(normalized_columns, techniques):
            if norm_col in normalised_df.columns:
                norm_values = eval(row[norm_col]) if isinstance(row[norm_col], str) else row[norm_col]

                for val in norm_values:
                    plot_data.append({
                        'technique': technique,
                        'normalized_f1': val
                    })

        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(
            data=plot_df,
            x='technique',
            y='normalized_f1',
            ax=ax
        )

        dataset_name = row.get('dataset_name', f'Dataset {idx + 1}')
        ax.set_title(f'{dataset_name}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Normalized F1 Score')
        ax.set_xlabel('Technique')
        ax.tick_params(axis='x', rotation=45, labelsize=8)

        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
        ax.legend(loc='upper right', fontsize=7)

    for j in range(num_datasets, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Normalized F1 Scores by Technique for Each Dataset', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output_path is not None:
        file_path = f"{output_path}\\meta_learners_performance_normalised_box_plots.png"
        fig.savefig(file_path, dpi=300)
        print(f"Saved box plot for normalised meta learners performance to {file_path}")

    for col in columns_to_keep:
        if col != 'dataset_name':
            normalised_df[f'{col.replace("_", " ")} mean'] = normalised_df[col].apply(
                lambda x: np.mean(eval(x)) if isinstance(x, str) else np.mean(x)
            )
            normalised_df[f'{col.replace("_", " ")} std'] = normalised_df[col].apply(
                lambda x: np.std(eval(x)) if isinstance(x, str) else np.std(x)
            )
            better_values = []
            for idx, row in normalised_df.iterrows():
                tech_values = eval(row[col]) if isinstance(row[col], str) else row[col]
                meta_values = eval(row["meta_learner_testing_f1_scores_normalized"]) if isinstance(row["meta_learner_testing_f1_scores_normalized"], str) else row["meta_learner_testing_f1_scores_normalized"]

                t_stat, p_value = ttest_rel(tech_values, meta_values)

                is_better = (p_value < 0.05) and (np.mean(tech_values) > np.mean(meta_values))

                better_values.append(is_better)

            normalised_df[f'{col.replace("_", " ")} better'] = better_values

    if output_path is not None:
        save_data_frame(normalised_df, f"{output_path}\\meta_learners_performance_normalised.csv")
        print(f"The meta learners performance normalise was save to {output_path}\\meta_learners_performance_normalised.csv")
    else:
        print(normalised_df)


def summaries_results(meta_learners_results, output_path):
    # Create summary dataframe with means and stds
    summary_df = meta_learners_results.copy()

    # Get all unique technique prefixes (excluding meta_learner)
    techniques = []
    for col in summary_df.columns:
        if '_testing_' in col:
            technique = col.rsplit('_testing_', 1)[0]
            if technique != 'meta_learner' and technique not in techniques:
                techniques.append(technique)

    # Metrics to process
    metrics = ['f1_scores']
    metrics_to_drop = ['loss', 'accuracies']
    for metric in metrics_to_drop:
        summary_df = summary_df.drop(
            columns=[col for col in summary_df.columns if metric in col]
        )

    for technique in techniques:
        for metric in metrics:
            tech_col = f'{technique}_testing_{metric}'
            meta_col = f'meta_learner_testing_{metric}'

            if tech_col in summary_df.columns and meta_col in summary_df.columns:
                summary_df[f'{tech_col.replace("_", " ")} mean'] = summary_df[tech_col].apply(
                    lambda x: np.mean(eval(x)) if isinstance(x, str) else np.mean(x)
                )
                summary_df[f'{tech_col.replace("_", " ")} std'] = summary_df[tech_col].apply(
                    lambda x: np.std(eval(x)) if isinstance(x, str) else np.std(x)
                )

                better_values = []
                for idx, row in summary_df.iterrows():
                    tech_values = eval(row[tech_col]) if isinstance(row[tech_col], str) else row[tech_col]
                    meta_values = eval(row[meta_col]) if isinstance(row[meta_col], str) else row[meta_col]

                    # Perform paired t-test
                    t_stat, p_value = ttest_rel(tech_values, meta_values)

                    # Determine if technique is statistically better (p < 0.05)
                    # For loss: lower is better, so tech < meta
                    # For accuracies/f1: higher is better, so tech > meta
                    if metric == 'loss':
                        is_better = (p_value < 0.05) and (np.mean(tech_values) < np.mean(meta_values))
                    else:  # accuracies or f1_scores
                        is_better = (p_value < 0.05) and (np.mean(tech_values) > np.mean(meta_values))

                    better_values.append(is_better)

                summary_df[f'{tech_col.replace("_", " ")} better'] = better_values

                summary_df = summary_df.drop(columns=[tech_col])

    if output_path is not None:
        save_data_frame(summary_df, f"{output_path}\\meta_learners_performance_summary.csv")
        print(f"The meta learners performance summary was save to {output_path}\\meta_learners_performance_summary.csv")
    else:
        print(summary_df)


def create_f1_comparison_heatmap_plot(row_dataframe, alpha, save_path):
    num_datasets = len(row_dataframe)
    num_cols = 3
    num_rows = int(np.ceil(num_datasets / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
    axes = np.array(axes).reshape(-1)

    for idx, (dataset_name, comparison_df) in enumerate(row_dataframe.items()):
        ax = axes[idx]

        sns.heatmap(
            comparison_df,
            annot=True,
            fmt='.0f',
            center=0,
            cmap="Blues",
            cbar=True,
            linewidths=0.5,
            linecolor='gray',
            vmin=-1,
            vmax=1,
            ax=ax
        )

        ax.set_title(f'{dataset_name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Technique (Compared Against)', fontsize=8)
        ax.set_ylabel('Technique', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)

    # Hide unused subplots
    for j in range(num_datasets, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f'F1 Score Comparison Heatmaps by Dataset\n(Paired T-test, α={alpha})',
        fontsize=14,
        fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(f"{save_path}//f1_comparison_heatmap_per_dataset.png", dpi=300, bbox_inches='tight')
        print(f'Saved figure of f1 comparison heatmap per dataset to {save_path}')
    plt.show()


def create_f1_comparison_heatmap(df: pd.DataFrame, alpha: float = 0.05,
                                 save_path: str = None):
    f1_columns = [col for col in df.columns if col.endswith('_testing_f1_scores')]

    technique_names = [col.replace('_testing_f1_scores', '') for col in f1_columns]

    n_techniques = len(technique_names)
    row_dataframe = {}
    total_comparison_matrix = np.zeros((n_techniques, n_techniques))
    for idx, row in df.iterrows():
        comparison_matrix = np.zeros((n_techniques, n_techniques))
        technique_scores = {}

        for f1_col, technique in zip(f1_columns, technique_names):
            f1_str = row[f1_col]
            if isinstance(f1_str, str):
                f1_str = f1_str.strip('[]')
                scores = [float(x.strip()) for x in f1_str.split(',')]
            else:
                scores = f1_str
            technique_scores[technique] = np.array(scores)

        for i, tech1 in enumerate(technique_names):
            for j, tech2 in enumerate(technique_names):
                if i == j:
                    comparison_matrix[i, j] += 0
                else:
                    scores1 = technique_scores[tech1]
                    scores2 = technique_scores[tech2]

                    t_stat, p_value = stats.ttest_rel(scores1, scores2)

                    if p_value < alpha:
                        if np.mean(scores1) > np.mean(scores2):
                            comparison_matrix[i, j] += 1
                            total_comparison_matrix[i, j] += 1
                        else:
                            comparison_matrix[i, j] += -1

        comparison_df = pd.DataFrame(
            comparison_matrix,
            index=technique_names,
            columns=technique_names
        )

        row_dataframe[row["dataset_name"]] = comparison_df


    total_comparison_df = pd.DataFrame(
        total_comparison_matrix,
        index=technique_names,
        columns=technique_names
    )

    create_f1_comparison_heatmap_plot(row_dataframe, alpha, save_path)

    # CREATE HEATMAP FOR TOTAL COMPARISON - ADD THIS SECTION
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        total_comparison_df,
        annot=True,
        fmt='.0f',
        center=0,
        cmap="Blues",  # Red-Yellow-Green colormap for better distinction
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        vmin=total_comparison_df.min().min(),
        vmax=total_comparison_df.max().max()
    )

    plt.title(f'Overall F1 Score Comparison Across All Datasets\n(Paired T-test, α={alpha})',
              fontsize=12, fontweight='bold')
    plt.xlabel('Technique (Compared Against)', fontsize=10)
    plt.ylabel('Technique', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}//f1_total_comparison_heatmap.png", dpi=300, bbox_inches='tight')
        print(f'Saved total comparison heatmap to {save_path}//f1_total_comparison_heatmap.png')

    plt.show()

    return comparison_df

def create_confusion_matrix(dataset, output_path, type):
    required_cols = [
        "model type",
        "technique",
        f"{type} true positives",
        f"{type} true negatives",
        f"{type} false positives",
        f"{type} false negatives",
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
    if output_path is not None:
        output_path = f"{output_path}\\{type}"
        os.makedirs(output_path, exist_ok=True)

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
            tp = float(module_df[f"{type} true positives"].iloc[0])
            tn = float(module_df[f"{type} true negatives"].iloc[0])
            fp = float(module_df[f"{type} false positives"].iloc[0])
            fn = float(module_df[f"{type} false negatives"].iloc[0])

            confusion_matrix = np.array([[tn, fp], [fn, tp]])

            sns.heatmap(confusion_matrix, annot=True, fmt=".0f",  cmap="Blues", ax=ax)
            ax.set_title(f"{model_type}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        # delete unused axes
        for counter in range(len(model_types), len(axes)):
            fig.delaxes(axes[counter])

        fig.suptitle(f"{type[0].upper()}{type[1:]} Confusion Matrices — Technique: {technique}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if output_path is not None:
            file_path = f"{output_path}\\confusion_matrices_{str(technique).replace(' ', '_')}.png"
            fig.savefig(file_path, dpi=300)
            print(f"Saved confusion matrices for technique '{technique}' to {file_path}")
        plt.show()

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
    if output_path is not None:
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
    if output_path is not None:
        file_path = f"{output_path}\\{metric_column_name.replace(" ","_")}_box_plot.png"
        print(f'Saved {metric_column_name}\'s box plot to {file_path}')
        plt.savefig(file_path, dpi=300)
    plt.show()

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

def create_meta_learner_comparison_boxplots(meta_learners_results, output_path, metric):
    print(f"Making meta-learner comparison boxplots for the {metric} metric")
    sns.set_style("darkgrid")

    df = meta_learners_results.copy()

    # Get all technique columns
    f1_columns = [col for col in df.columns if col.endswith(f'_testing_{metric.replace(" ","_")}')]

    # Extract technique names from column names
    techniques = [col.replace(f'_testing_{metric.replace(" ","_")}', '') for col in f1_columns]

    # Number of datasets (rows)
    num_datasets = len(df)

    # Create subplots - one per dataset
    num_cols = 3
    num_rows = int(np.ceil(num_datasets / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 5))
    axes = np.array(axes).reshape(-1)

    # Process each dataset (row)
    for idx, (row_idx, row) in enumerate(df.iterrows()):
        ax = axes[idx]

        # Prepare data for boxplot
        plot_data = []

        for f1_col, technique in zip(f1_columns, techniques):
            if f1_col in row.index and pd.notna(row[f1_col]):
                # Convert string representation to list if needed
                f1_values = eval(row[f1_col]) if isinstance(row[f1_col], str) else row[f1_col]

                # Add each value with its technique label
                for val in f1_values:
                    plot_data.append({
                        'technique': technique,
                        metric.replace(" ","_"): val
                    })

        # Convert to DataFrame for seaborn
        plot_df = pd.DataFrame(plot_data)

        # Create boxplot
        sns.boxplot(
            data=plot_df,
            x='technique',
            y=metric.replace(" ","_"),
            ax=ax
        )

        # Set title (use dataset_name if available)
        dataset_name = row.get('dataset_name', f'Dataset {idx + 1}')
        ax.set_title(f'{dataset_name}')
        ax.set_ylabel(metric)
        ax.set_xlabel('Technique')
        ax.tick_params(axis='x', rotation=90)

        # Adjust y-axis limits for better visualization
        ax.set_ylim([0, 1.05])

    # Hide any unused subplots
    for j in range(num_datasets, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'{metric.replace(" ","_")} Comparison Across Techniques', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output_path is not None:
        file_path = f"{output_path}\\{metric.replace(" ","_")}_comparison_boxplots.png"
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f'Saved {metric} comparison boxplots to {file_path}')

    plt.show()