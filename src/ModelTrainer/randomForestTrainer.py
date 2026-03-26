import random
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from src.Utils.constants import *
from src.Utils.datasetHandler import prepared_meta_feature_dataset
from src.Utils.fileHandler import load_settings, folder_maker
from src.Utils.metaLearnerStatsCalculator import TrainingMetaLearnerStats, TestingMetaLearnerStats


def training_meta_random_forests(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    random.seed(seed)
    for target_column in META_LEANER_TARGET_COLUMNS:
        print(f"Training random forests for { target_column.replace("_"," ")}...")
        cleaned_training_set = prepared_meta_feature_dataset(training_set,target_column,False)
        cleaned_testing_set = prepared_meta_feature_dataset(testing_set,target_column,False)
        training_result = train_meta_random_forest(settings[target_column],
                                                   cleaned_training_set,
                                                   cleaned_testing_set,
                                                   seed,
                                                   "n.a",
                                                   kFold)
        seed = random.randint(0, 4294967295)
        testing_result = train_meta_random_forest(settings[target_column],
                                                  cleaned_training_set,
                                                  cleaned_testing_set,
                                                  seed,
                                                  target_column,
                                                  0)
        seed = random.randint(0, 4294967295)
        result = {
            "model type": "Random forest",
            "technique": target_column.replace("_"," "),
            **training_result,
            **testing_result
        }
        results.append(result)
    return results

def train_meta_random_forest(params, training_set, testing_set, seed, target_column ='na', kFold = 5):
    training_x = training_set[0]
    training_y = training_set[1].to_numpy()
    testing_x = testing_set[0]
    testing_y = testing_set[1]

    if target_column != 'na':
        folder_path = f"{MODULE_PATH}RandomForest\\{datetime.now().strftime("%Y%m%d_%h")}"
        folder_maker(folder_path)

    rf_params = params.copy()
    if rf_params.get("bootstrap") is False:
        rf_params["max_samples"] = None

    if kFold == 0:
        random_forests_stats = TestingMetaLearnerStats()

        forest = RandomForestClassifier(random_state=seed, **rf_params)
        forest.fit(training_x, training_y)
        y_test_pred = forest.predict(testing_x)

        random_forests_stats.update_stats(testing_y, y_test_pred)

        if target_column != 'na':
            joblib.dump(forest, f'{folder_path}\\{target_column}.pkl')
    else:
        kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)

        random_forests_stats = TrainingMetaLearnerStats()

        counter = 1

        for train_idx, test_idx in kf.split(training_x):
            x_train = training_x[train_idx]
            y_train = training_y[train_idx]

            forest = RandomForestClassifier(random_state=seed, **rf_params)
            forest.fit(x_train, y_train)

            y_train_pred = forest.predict(x_train)
            y_test_pred = forest.predict(testing_x)

            random_forests_stats.update_stats(y_train, y_train_pred, testing_y, y_test_pred)

            if target_column != 'na':
                joblib.dump(forest, f'{folder_path}\\{target_column}_fold_{counter}.pkl')

            counter = counter + 1

    return random_forests_stats.get_stats_json_object()