import ast
import random
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import torch
from numpy import mean
from scipy.stats import ttest_ind, friedmanchisquare, studentized_range

from src.Models.NN.network import Network
from src.Utils.constants import TEST_TYPES
from src.Utils.menus import show_menu


class MetaLearner():
    _instance = None
    models_for_each_technique = {}
    techniques = []
    model_types = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, meta_learners_results):
        self.techniques = list(meta_learners_results["technique"].dropna().unique())
        self.model_types = list(meta_learners_results["model type"].dropna().unique())
        options = TEST_TYPES
        options.append("One model")

        test_type = show_menu("Select the test method which will be used to rank the techniques: ", options)
        if test_type == TEST_TYPES[0]:
            self.get_best_models_base_on_mann_whitney(meta_learners_results)
        elif test_type == TEST_TYPES[1]:
            self.get_best_models_base_on_friedman(meta_learners_results)
        else:
            model_type = show_menu("Select the model type which will be used: ", self.model_types)
            self.get_model_info(meta_learners_results, model_type)
        print("Meta-Learner:")
        for technique in  self.techniques:
            types =[]
            for model in self.models_for_each_technique[technique]:
                types.append(model['type'])
            print(f"Technique: {technique}, Model Types: {types}")

    def predict_best_technique(self, meta_features):
        techniques_predicted = {}
        for technique in  self.techniques:
            techniques_predicted[technique] = 0
            best_mean_metric = -1
            best_metric = []
            for model_object in self.models_for_each_technique[technique]:
                if model_object['type'] == "Neural Network":
                    checkpoint = torch.load(model_object['path'])
                    model = Network(**checkpoint["model_kwargs"])
                    model.load_state_dict(checkpoint["state_dict"])
                    model.eval()

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = model.to(device)

                    input_np = meta_features.to_numpy(dtype="float32", copy=False)
                    input_data = torch.from_numpy(input_np).to(device)
                    with torch.no_grad():
                        is_best = model(input_data)
                        if is_best[0][1] == 1.0:
                            techniques_predicted[technique] += 1
                else:
                    model = joblib.load(model_object['path'])
                    is_best = model.predict(meta_features)
                    if model_object['type'] == "svm":
                        if is_best[0] == 1:
                            techniques_predicted[technique] += 1
                    else:
                        if is_best[0][1]:
                            techniques_predicted[technique] += 1
                mean_metric = mean(model_object['metric'])
                if mean_metric > best_mean_metric:
                    best_mean_metric = mean_metric
                    best_metric = model_object['metric']
            techniques_predicted[technique] = techniques_predicted[technique]/len(self.models_for_each_technique[technique])
            techniques_predicted[technique] = {
                'isBested': 1 if techniques_predicted[technique]  >= 0.5 else 0,
                'metric': best_metric
            }
        best_metric = -1
        best_metrics = []
        best_technique = []
        for technique in  self.techniques:
            if techniques_predicted[technique]['isBested'] == 1:
                if best_metric == -1:
                    best_metrics = techniques_predicted[technique]['metric']
                    best_metric = mean(best_metrics)
                    best_technique = [technique]
                else:
                    stat, p_value = ttest_ind(techniques_predicted[technique]['metric'], best_metrics, equal_var=False)
                    metric = mean(techniques_predicted[technique]['metric'])
                    if metric > best_metric and p_value < 0.05:
                        best_metrics = techniques_predicted[technique]['metric']
                        best_metric = mean(best_metrics)
                        best_technique = [technique]
                    elif p_value >= 0.05:
                        best_technique.append(technique)

        return best_technique[random.randint(0, len(best_technique) - 1)]

    def get_best_models_base_on_mann_whitney(self, meta_learners_results):
        for technique in self.techniques:
            results_per_technique = meta_learners_results[
                meta_learners_results["technique"].replace(" ", "_") == technique]
            best_metric = -1
            best_model_types = []
            best_f1_scores = None
            for model_type in self.model_types:
                results_per_technique_and_model = results_per_technique[
                    results_per_technique["model type"] == model_type]
                if not results_per_technique_and_model.empty:
                    f1_scores = results_per_technique_and_model["testing f1"].iloc[0]
                    f1_scores = ast.literal_eval(f1_scores)
                    metric = mean(f1_scores)
                    if best_metric != -1:
                        stat, p_value = ttest_ind(f1_scores, best_f1_scores, equal_var=False)
                        if metric > best_metric and p_value < 0.05:
                            best_metric = metric
                            best_model_types = [{
                                "type": model_type,
                                "metric": f1_scores,
                                "path": results_per_technique_and_model["model path"].values[0]
                            }]
                            best_f1_scores = f1_scores
                        elif p_value >= 0.05:
                            best_model_types.append({
                                "type": model_type,
                                "metric": f1_scores,
                                "path": results_per_technique_and_model["model path"].values[0]
                            })
                    else:
                        best_metric = metric
                        best_model_types = [{
                            "type": model_type,
                            "metric": f1_scores,
                            "path": results_per_technique_and_model["model path"].values[0]
                        }]
                        best_f1_scores = f1_scores
            self.models_for_each_technique[technique] = best_model_types

    def get_best_models_base_on_friedman(self, meta_learners_results):
        meta_learners_results = meta_learners_results[["technique", "model type", "testing loses", "model path"]]
        ranked_frames = {}
        alpha = 0.1
        for technique in self.techniques:
            results_per_technique = meta_learners_results[
                meta_learners_results["technique"].replace(" ", "_") == technique
                ].reset_index(drop=True)

            if results_per_technique.empty:
                continue

            losses_by_row = [
                ast.literal_eval(losses)
                for losses in results_per_technique["testing loses"]
            ]

            run_count = len(losses_by_row[0])
            ranked_rows = {}
            for model_type in self.model_types:
                ranked_rows[model_type] = [0] * run_count

            for score_idx in range(run_count):
                column_values = [row[score_idx] for row in losses_by_row]
                sorted_row_indices = sorted(
                    range(len(column_values)),
                    key=lambda i: column_values[i],
                    reverse=True
                )

                for rank, row_idx in enumerate(sorted_row_indices, start=1):
                    ranked_rows[self.model_types[row_idx]][score_idx] = rank

            # 2) Friedman test across all samples
            statistic, p_value = friedmanchisquare(
                *[ranked_rows[model_type] for model_type in self.model_types]
            )

            for model_type in self.model_types:
                if p_value > alpha:
                    ranked_rows[model_type] = 1
                else:
                    ranked_rows[model_type] = round(np.mean(ranked_rows[model_type]))

            # 3) Nemenyi pairwise comparisons
            if p_value <= alpha:
                if studentized_range is not None:
                    number_of_models = len(self.model_types)
                    q_critical = studentized_range.ppf(1 - alpha, number_of_models, np.inf) / np.sqrt(2)
                    critical_difference = q_critical * np.sqrt(
                        number_of_models * (number_of_models - 1) / (6 * run_count)
                    )
                else:
                    critical_difference = 0.0

                changed = True
                while changed:
                    changed = False
                    for model_type1, model_type2 in combinations(self.model_types, 2):
                        diff = abs(ranked_rows[model_type1] - ranked_rows[model_type2])
                        significant = diff > critical_difference if critical_difference > 0 else False
                        if not significant:
                            if ranked_rows[model_type1] < ranked_rows[model_type2]:
                                ranked_rows[model_type2] = ranked_rows[model_type1]
                                changed = True
                            elif ranked_rows[model_type2] < ranked_rows[model_type1]:
                                ranked_rows[model_type1] = ranked_rows[model_type2]
                                changed = True

            valid_ranks = sorted({rank for rank in ranked_rows.values() if rank >= 1})
            rank_map = {rank: index for index, rank in enumerate(valid_ranks, start=1)}
            ranked_rows = {
                model_type: rank_map[rank] if rank >= 1 else rank
                for model_type, rank in ranked_rows.items()
            }

            best_models = {
                model_type: 1 if rank == 1 else 0
                for model_type, rank in ranked_rows.items()
            }

            ranked_frames[technique] = best_models

        ranked_results = pd.DataFrame(
            [
                {"technique": technique, **ranked_rows}
                for technique, ranked_rows in ranked_frames.items()
            ]
        )
        for _, row in ranked_results.iterrows():
            technique = row["technique"]
            best_model_types = []
            for model_type in self.model_types:
                if row[model_type] == 1:
                    results_per_technique_and_model = meta_learners_results[
                        (meta_learners_results["technique"].replace(" ", "_") == technique) &
                        (meta_learners_results["model type"] == model_type)
                    ]
                    f1_scores = results_per_technique_and_model["testing loses"].iloc[0]
                    f1_scores = ast.literal_eval(f1_scores)
                    best_model_types.append({
                        "type": model_type,
                        "metric": f1_scores,
                        "path": results_per_technique_and_model["model path"].values[0]
                    })
            self.models_for_each_technique[technique] = best_model_types

    def get_model_info(self, meta_learners_results, model_type):
        for technique in self.techniques:
            results_per_technique = meta_learners_results[meta_learners_results["technique"].replace(" ", "_") == technique]
            results_per_technique_and_model = results_per_technique[results_per_technique["model type"] == model_type]
            f1_scores = results_per_technique_and_model["testing f1"].iloc[0]
            f1_scores = ast.literal_eval(f1_scores)
            self.models_for_each_technique[technique] = [{
                "type": model_type,
                "metric": f1_scores,
                "path": results_per_technique_and_model["model path"].values[0]
            }]