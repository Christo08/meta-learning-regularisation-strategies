import ast
import random
from statistics import mean

from scipy.stats import ttest_ind

from src.ModelTrainer.decisionTreeTrainer import training_meta_decision_trees
from src.ModelTrainer.knnTrainer import training_meta_k_nearest_neighbors
from src.ModelTrainer.nnTrainer import training_meta_nns, train_basic_nns
from src.ModelTrainer.randomForestTrainer import training_meta_random_forests
from src.ModelTrainer.svmTrainer import training_meta_support_vector_machines
from src.Utils.datasetHandler import load_subset
from src.Utils.datasetSettingHandler import DatasetsSettingsHandler
from src.Utils.fileHandler import *
from src.Utils.menus import show_meta_leaner_type_menu


def train_meta_learners(training_set, validation_set):
    training_set.drop(columns=["dataset_name","file_name"], inplace=True)
    validation_set.drop(columns=["dataset_name", "file_name"], inplace=True)
    selected_meta_learn_types = show_meta_leaner_type_menu()
    number_of_folds = int(input("How many folds do you want the meta-learner to get trained? "))
    results = pd.DataFrame(columns=["model type", "technique",  "training loses", "testing loses"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings_file_path = input(f"Enter the path of the settings index file: ")
    output_path = input("Enter the path of the output stats folder: ")
    settings = load_json_file(settings_file_path)

    for selected_meta_learn_type in selected_meta_learn_types:
        seed = random.randint(0, 4294967295)
        if selected_meta_learn_type == META_LEARN_TYPES[1]:
            settings_file_path = settings["DecisionTrees"]
            result = training_meta_decision_trees(settings_file_path, training_set, validation_set, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[4]:
            settings_file_path = settings["RandomForest"]
            result = training_meta_random_forests(settings_file_path, training_set, validation_set, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[2]:
            settings_file_path = settings["KNearestNeighbors"]
            result = training_meta_k_nearest_neighbors(settings_file_path, training_set, validation_set, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[3]:
            settings_file_path = settings["NeuralNetworks"]
            result = training_meta_nns(settings_file_path, training_set, validation_set, seed, number_of_folds)
        elif selected_meta_learn_type == META_LEARN_TYPES[5]:
            settings_file_path = settings["SupportVectorMachines"]
            result = training_meta_support_vector_machines(settings_file_path, training_set, validation_set, seed, number_of_folds)
        else:
            return
        results = pd.concat([results, pd.DataFrame(result)], ignore_index=True)
    output_path = f'{output_path}\\{timestamp}'
    folder_maker(output_path)
    file_name = f"{output_path}\\meta_learners_results.csv"
    save_data_frame(results, file_name)

def test_meta_learner(subsets, ranked_subsets, output_path, meta_learner):
    seed = random.randint(0, 4294967295)
    random.seed(seed)
    columns_to_drop = ["dataset_name", "file_name"] + TARGET_COLUMNS
    details = []
    print("Predict best techniques for each dataset...")
    for _, subset in subsets.iterrows():
        meta_features = subset.drop(labels=columns_to_drop, errors='ignore')

        best_technique = meta_learner.predict_best_technique(meta_features.to_frame().T)
        details.append({
            "dataset_name": subset["dataset_name"],
            "best_technique": best_technique,
            "file_path": subset["file_name"]
        })
    generate_performs = input("Do you want to generate the performs of the basic NN (Y/N)?").upper() == "Y"
    results = pd.DataFrame()
    if generate_performs:
        number_of_folds = int(input("How many folds do you want to use? "))
        dataset_settings_handler = DatasetsSettingsHandler()
        for detail in details:
            seed = random.randint(0, 4294967295)
            nn_settings = get_latest_nn_settings(detail["dataset_name"])
            training_set, testing_set, category_columns = load_subset(detail["file_path"],
                                                                      seed,
                                                                      dataset_settings_handler.get_dataset_by_name(detail["dataset_name"]))
            seed = random.randint(0, 4294967295)
            instance_json_object = train_nns(detail["dataset_name"],
                                             detail["best_technique"],
                                             seed,
                                             training_set,
                                             testing_set,
                                             category_columns,
                                             number_of_folds,
                                             nn_settings)
            results = pd.concat([results, pd.DataFrame([instance_json_object])], ignore_index=True)
            save_data_frame(results, output_path)
    else:
        dataset = load_meta_features_csv()
        for detail in details:
            try:
                if str(detail['file_path']) == "nan":
                    first_match = dataset[(dataset["dataset_name"] == detail["dataset_name"]) &
                                          (dataset["subset_type"] == "full")].iloc[0]
                    first_ranked_match = ranked_subsets[(ranked_subsets["dataset_name"] == detail["dataset_name"])  &
                                                        (ranked_subsets["file_name"].isna())].iloc[0]
                else:
                    first_match = dataset[(dataset["file_name"] == detail["file_path"])].iloc[0]
                    first_ranked_match = ranked_subsets[(ranked_subsets["file_name"] == detail["file_path"])].iloc[0]
            except IndexError:
                print(f"No match found for file: {detail['file_path']}")
                continue
            instance_json_object = {
                "dataset_name": detail["dataset_name"],
                "seed": first_match["seed"],
                "best_technique": detail["best_technique"]
            }
            print(f"Dataset name: {detail["dataset_name"]}")
            print(f"Best technique: {detail["best_technique"]}")
            for config in REGULARISATION_TECHNIQUES:
                instance_json_object[f"{config['fileName']}_training_loss"] = first_match[f"{config['fileName']}_training_loss"]
                instance_json_object[f"{config['fileName']}_training_accuracies"] = first_match[f"{config['fileName']}_training_accuracies"]
                instance_json_object[f"{config['fileName']}_training_f1_scores"] = first_match[f"{config['fileName']}_training_f1_scores"]
                instance_json_object[f"{config['fileName']}_testing_loss"] = first_match[f"{config['fileName']}_testing_loss"]
                instance_json_object[f"{config['fileName']}_testing_accuracies"] = first_match[f"{config['fileName']}_testing_accuracies"]
                instance_json_object[f"{config['fileName']}_testing_f1_scores"] = first_match[f"{config['fileName']}_testing_f1_scores"]
                instance_json_object[f"{config['fileName']}_rank"] = first_ranked_match[f"{config['fileName']}"]
                if config['fileName'].replace("_"," ") == detail["best_technique"]:
                    instance_json_object["meta_learner_training_loss"] = first_match[f"{config['fileName']}_training_loss"]
                    instance_json_object["meta_learner_training_accuracies"] = first_match[f"{config['fileName']}_training_accuracies"]
                    instance_json_object["meta_learner_training_f1_scores"] = first_match[f"{config['fileName']}_training_f1_scores"]
                    instance_json_object["meta_learner_testing_loss"] = first_match[f"{config['fileName']}_testing_loss"]
                    instance_json_object["meta_learner_testing_accuracies"] = first_match[f"{config['fileName']}_testing_accuracies"]
                    instance_json_object["meta_learner_testing_f1_scores"] = first_match[f"{config['fileName']}_testing_f1_scores"]
                    instance_json_object[f"meta_learner_rank"] = first_ranked_match[f"{config['fileName']}"]
            results = pd.concat([results, pd.DataFrame([instance_json_object])], ignore_index=True)
        save_data_frame(results, output_path)

def create_meta_learner(meta_learners_results, meta_learners_results_per_technique):
    techniques = list(meta_learners_results["technique"].dropna().unique())
    model_types = list(meta_learners_results["model type"].dropna().unique())
    for technique in techniques:
        meta_learners_results_per_technique = meta_learners_results[
            meta_learners_results["technique"].replace(" ", "_") == technique]
        best_metric = -1
        best_model_types = []
        best_f1_scores = None
        for model_type in model_types:
            meta_learners_results_per_technique_and_model = meta_learners_results_per_technique[
                meta_learners_results_per_technique["model type"] == model_type]
            if not meta_learners_results_per_technique_and_model.empty:
                f1_scores = meta_learners_results_per_technique_and_model["testing f1"].iloc[0]
                f1_scores = ast.literal_eval(f1_scores)
                metric = mean(f1_scores)
                if best_metric != -1:
                    stat, p_value = ttest_ind(f1_scores, best_f1_scores, equal_var=False)
                    if metric > best_metric and p_value < 0.05:
                        best_metric = metric
                        best_model_types = [model_type]
                        best_f1_scores = f1_scores
                    elif p_value >= 0.05:
                        best_model_types.append(model_type)
                else:
                    best_metric = metric
                    best_model_types = [model_type]
                    best_f1_scores = f1_scores

def train_nns(dataset_name, best_technique, seed, training_set, testing_set, category_columns, number_of_folds, nn_settings):
    print("")
    print("Dataset name: " + dataset_name)
    print("Seed: " + str(seed))
    print("Predict best technique: " +best_technique)
    # Add dataset name, seed and meta feature
    instance_json_object = {
        "dataset_name": dataset_name,
        "seed": seed,
        "best_technique": best_technique
    }
    for config in REGULARISATION_TECHNIQUES:
        matrices, dynamics_meta_learners = train_basic_nns(nn_settings,
                                                           config["param"],
                                                           training_set,
                                                           testing_set,
                                                           seed,
                                                           category_columns,
                                                           number_of_folds)

        if config["name"] == "baseline":
            instance_json_object = {**instance_json_object, **dynamics_meta_learners}

        instance_json_object[f"{config['fileName']}_training_loss"] = matrices["training_loss"]
        instance_json_object[f"{config['fileName']}_training_accuracies"] = matrices["training_accuracies"]
        instance_json_object[f"{config['fileName']}_training_f1_scores"] = matrices["training_f1_scores"]
        instance_json_object[f"{config['fileName']}_testing_loss"] = matrices["testing_loss"]
        instance_json_object[f"{config['fileName']}_testing_accuracies"] = matrices["testing_accuracies"]
        instance_json_object[f"{config['fileName']}_testing_f1_scores"] = matrices["testing_f1_scores"]

    instance_json_object["meta_learner_training_loss"] =  instance_json_object[f"{best_technique}_training_loss"]
    instance_json_object["meta_learner_training_accuracies"] = instance_json_object[f"{best_technique}_training_accuracies"]
    instance_json_object["meta_learner_training_f1_scores"] = instance_json_object[f"{best_technique}_training_f1_scores"]
    instance_json_object["meta_learner_testing_loss"] = instance_json_object[f"{best_technique}_testing_loss"]
    instance_json_object["meta_learner_testing_accuracies"] = instance_json_object[f"{best_technique}_testing_accuracies"]
    instance_json_object["meta_learner_testing_f1_scores"] = instance_json_object[f"{best_technique}_testing_f1_scores"]

    return instance_json_object