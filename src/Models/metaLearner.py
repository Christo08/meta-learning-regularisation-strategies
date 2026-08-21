import ast
import random

import joblib
import torch
from numpy import mean
from scipy.stats import ttest_ind

from src.Models.NN.network import Network


class MetaLearner():
    _instance = None
    models_for_each_technique = {}
    techniques = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, meta_learners_results):
        self.techniques = list(meta_learners_results["technique"].dropna().unique())
        model_types = list(meta_learners_results["model type"].dropna().unique())
        for technique in self.techniques:
            results_per_technique = meta_learners_results[meta_learners_results["technique"].replace(" ", "_") == technique]
            best_metric = -1
            best_model_types = []
            best_f1_scores = None
            for model_type in model_types:
                results_per_technique_and_model = results_per_technique[results_per_technique["model type"] == model_type]
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