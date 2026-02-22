import math
import random
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from pmlb import fetch_data
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

from Utils.fileHandler import save_subset
from Utils.metaFeatureCalculator import calculate_meta_features

dataset = pd.DataFrame()
dataset_name = ""
category_columns = []

# Constants
MIN_CLASSES_REQUIRED = 2
MIN_INSTANCES_PER_SUBSET = 100
MIN_FEATURE_FRACTION = 0.5
OFFSET_RANGE_START = 1

def create_subsets_with_seeds(database_name, number_of_subsets_need, class_seeds, features_seeds, instances_seeds, dataset_settings):
    print("Recreating " + str(number_of_subsets_need) + " Subsets for the " + database_name + " dataset")
    dataset = load_raw_dataset(dataset_settings)
    dataset = clean_dataset(dataset)
    numeric_data = dataset.select_dtypes(include=[np.number])
    assert not np.isinf(numeric_data.values).any(), "Inf in numeric input DataFrame"

    class_subsets, class_seeds = make_classes_subsets(dataset, number_of_subsets_need, class_seeds)

    features_subsets, features_seeds = make_features_subsets(dataset, len(features_seeds), features_seeds)

    instances_subsets, instances_seeds = make_instances_subsets(dataset, len(instances_seeds), instances_seeds)

    subsets = class_subsets + features_subsets + instances_subsets
    seeds = class_seeds + features_seeds + instances_seeds

    subsets = np.array(subsets, dtype=object)

    return_subsets = []
    meta_features = []
    subset_file_paths = []
    subsets_category_columns = []
    for subset, seed in zip(subsets, seeds):
        subset, subset_category_columns = encode_categories_features(subset, dataset_settings['categoryColumns'])
        subset = remap_targets(subset)

        meta_features.append(calculate_meta_features(subset, dataset_settings['categoryColumns']))

        subset = normalise(subset, subset_category_columns,["target"])

        subset_file_paths.append(save_subset(subset, seed["seed"], database_name))

        subset.reset_index(drop=True, inplace=True)
        return_subsets.append(subset)
        subsets_category_columns.append(subset_category_columns)

    return return_subsets, meta_features, seeds, subsets_category_columns, subset_file_paths

def load_subset(file_path, seed, dataset_settings):
    subset = pd.read_csv(file_path)
    training_set, testing_set = splitSet(subset, seed)
    full_category_columns = dataset_settings['categoryColumns']
    category_columns = []
    for categoryColumn in full_category_columns:
        if categoryColumn in subset.columns:
            category_columns.append(categoryColumn)

    return training_set, testing_set, category_columns

def create_subsets(database_name, number_of_subsets_need, dataset_settings, need_split=True):
    print("Creating " + str(number_of_subsets_need) + " Subsets for the " + database_name + " dataset")
    dataset = load_raw_dataset(dataset_settings)
    dataset = clean_dataset(dataset)
    numeric_data = dataset.select_dtypes(include=[np.number])
    assert not np.isinf(numeric_data.values).any(), "Inf in numeric input DataFrame"

    class_subsets, class_seeds = make_classes_subsets(dataset, number_of_subsets_need)
    number_of_feature_subset = (number_of_subsets_need - len(class_subsets)) // 2

    features_subsets, features_seeds = make_features_subsets(dataset, number_of_feature_subset)
    number_of_instance_subset = number_of_subsets_need - len(features_subsets) - len(class_subsets)

    instances_subsets, instances_seeds = make_instances_subsets(dataset, number_of_instance_subset)

    subsets = class_subsets + features_subsets + instances_subsets
    seeds = class_seeds + features_seeds + instances_seeds

    subsets = np.array(subsets, dtype=object)

    return_subsets = []
    training_sets = []
    testing_sets = []
    meta_features = []
    subsets_category_columns = []
    subset_file_paths = []
    for subset, seed in zip(subsets, seeds):
        subset, subset_category_columns = encode_categories_features(subset, dataset_settings['categoryColumns'])
        subset = remap_targets(subset)

        meta_features.append(calculate_meta_features(subset, dataset_settings['categoryColumns']))

        subset = normalise(subset, subset_category_columns,["target"])

        subset.reset_index(drop=True, inplace=True)

        subset_file_paths.append(save_subset(subset, seed["seed"], database_name))

        if need_split:
            training_set, testing_set = splitSet(subset, seed["seed"])

            training_sets.append(training_set)
            testing_sets.append(testing_set)
        else:
            return_subsets.append(subset)
        subsets_category_columns.append(subset_category_columns)

    if need_split:
        return training_sets, testing_sets, meta_features, seeds, subsets_category_columns, subset_file_paths
    else:
        return return_subsets, meta_features, seeds, subsets_category_columns, subset_file_paths

def load_dataset(dataset_settings):
    dataset = load_raw_dataset(dataset_settings)
    dataset = clean_dataset(dataset)
    numeric_data = dataset.select_dtypes(include=[np.number])
    assert not np.isinf(numeric_data.values).any(), "Inf in numeric input DataFrame"

    seed = random.randint(1, 100000)
    random.seed(seed)

    dataset, dataset_category_columns = encode_categories_features(dataset, dataset_settings['categoryColumns'])
    meta_features = calculate_meta_features(dataset, dataset_settings['categoryColumns'])

    dataset = normalise(dataset, dataset_category_columns, ["target"])

    dataset = remap_targets(dataset)

    dataset.reset_index(drop=True, inplace=True)

    training_set, testing_set = splitSet(dataset, seed)

    return [training_set], [testing_set], [meta_features], [seed], [dataset_category_columns]

def load_optimiser_dataset(seed, dataset_settings):
    dataset = load_raw_dataset(dataset_settings)
    dataset = clean_dataset(dataset)

    target_columns = [col for col in dataset.columns if 'target' in col]
    dataset = normalise(dataset, dataset_settings['categoryColumns'], target_columns)
    dataset, category_columns  = encode_categories_features(dataset, dataset_settings['categoryColumns'])

    return splitSet(dataset, seed), category_columns

def apply_smote(x, y, seed, number_of_neighbors, category_columns):
    if isinstance(y, pd.DataFrame):
        y = y.idxmax(axis=1).str.extract(r'(\d+)$').astype(int).squeeze()

    class_counts = Counter(y.values.ravel())
    min_class_samples = min(class_counts.values())
    if min_class_samples <= 1:
        raise ValueError("Cannot apply smote.")

    safe_neighbors = min(number_of_neighbors, min_class_samples - 1)
    if safe_neighbors < 1:
        raise ValueError("Cannot apply smote.")

    if not category_columns:
        oversample = SMOTE(random_state=seed, k_neighbors=safe_neighbors)
    else:
        category_indices = [x.columns.get_loc(col) for col in category_columns]
        oversample = SMOTENC(random_state=seed, categorical_features=category_indices, k_neighbors=safe_neighbors)

    assert x.shape[0] == y.shape[0], "Mismatched number of samples between X and Y before applying SMOTE."

    x_resampled, y_resampled = oversample.fit_resample(x, y)
    assert x.shape[0] == y.shape[0], "Mismatched number of samples between X and Y after applying SMOTE."

    encoded_y = pd.get_dummies(y_resampled).astype(int)

    return x_resampled, encoded_y

#helper function
def make_classes_subsets(dataset, number_of_subsets_need, seeds=None):
    if seeds is None:
        seeds = []
    subsets = []
    class_labels = dataset['target'].unique().tolist()
    number_of_classes = len(class_labels)
    if number_of_classes <= MIN_CLASSES_REQUIRED:
        return subsets, seeds

    number_of_class_subset = number_of_subsets_need // 3
    number_of_unique_classes_combos = sum(math.comb(number_of_classes, k) for k in range(1, number_of_classes - MIN_CLASSES_REQUIRED)) - 1
    counter = 0
    used_class_combos = set()
    new_seeds =  len(seeds) < number_of_class_subset
    while counter < number_of_unique_classes_combos and len(subsets) < number_of_class_subset:
        if new_seeds:
            seed = random.randint(1, 100000)
        else:
            seed = seeds[counter]["seed"]

        combo_size = random.randint(1, number_of_classes - MIN_CLASSES_REQUIRED)
        combo = tuple(sorted(random.sample(class_labels, combo_size)))

        if combo in used_class_combos:
            continue

        used_class_combos.add(combo)
        subset = dataset[~dataset['target'].isin(combo)]

        if len(subset) < MIN_INSTANCES_PER_SUBSET:
            continue

        subsets.append(subset.copy())
        if new_seeds:
            seeds.append({"seed": seed, "subsetType": "classes"})

        counter += 1

    return subsets, seeds

def make_features_subsets(dataset, number_of_feature_subset, seeds=None):
    if seeds is None:
        seeds = []
    subsets = []
    features = [col for col in dataset.columns if col != 'target']

    number_of_features = len(features)
    maximum_subset_size = round(number_of_features * MIN_FEATURE_FRACTION)

    counter = 0
    number_of_unique_feature_combos = sum(math.comb(number_of_features, k) for k in range(1, maximum_subset_size)) - 1

    used_features_combos = set()
    new_seeds = len(seeds) < number_of_feature_subset
    while counter < number_of_unique_feature_combos and len(subsets) < number_of_feature_subset:
        if new_seeds:
            seed = random.randint(1, 100000)
        else:
            seed = seeds[counter]["seed"]
        random.seed(seed)

        combo_size = random.randint(1, maximum_subset_size)
        combo = tuple(sorted(random.sample(features, combo_size)))

        if combo in used_features_combos:
            continue

        subset = dataset.drop(columns=list(combo))

        subsets.append(subset.copy())
        if new_seeds:
            seeds.append({"seed": seed, "subsetType": "features"})

        counter += 1

    return subsets,seeds

def make_instances_subsets(dataset, number_of_instances_subsets_needed, seeds=None):
    if seeds is None:
        seeds = []
    subsets = []
    number_of_instances = len(dataset)
    interval_size = (number_of_instances - MIN_INSTANCES_PER_SUBSET) // number_of_instances_subsets_needed
    new_seeds = len(seeds) < number_of_instances_subsets_needed

    for counter in range(number_of_instances_subsets_needed):
        if new_seeds:
            seed = random.randint(1, 100000)
        else:
            seed = seeds[counter]["seed"]

        offset = random.randint(OFFSET_RANGE_START, interval_size)
        subset_size = MIN_INSTANCES_PER_SUBSET + interval_size * counter + offset
        subset_size = min(subset_size, math.floor(number_of_instances-dataset.shape[0]*0.1))
        create_subset = True
        while create_subset:
            subset, _ = train_test_split(
                dataset,
                train_size=subset_size,
                stratify=dataset['target'],
                random_state=seed
            )
            create_subset = False
            instance_per_classes = subset['target'].value_counts()
            for target, count in instance_per_classes.items():
                if count < 10:
                    if len(subset['target'].unique()) > MIN_CLASSES_REQUIRED:
                        subset = subset[~subset['target'].isin([target])]
                    else:
                        create_subset = True
                        break


        subsets.append(subset.copy())
        if new_seeds:
            seeds.append({"seed": seed, "subsetType": "instances"})

    return subsets, seeds

def load_raw_dataset(dataset_settings):
    if dataset_settings["type"] == "csv":
        dataset = pd.read_csv(dataset_settings["filePath"])
    else:
        dataset = fetch_data(dataset_settings["pmlbName"])

    if not(dataset_settings["targetColumn"] == "target"):
        dataset.rename(columns={dataset_settings["targetColumn"]: "target"}, inplace=True)

    if not(len(dataset_settings["dropColumns"]) == 0):
        dataset = dataset.drop(columns=dataset_settings["dropColumns"])

    dataset['target'] = dataset['target'].astype('category')
    dataset['target'] = dataset['target'].cat.codes
    return dataset

def clean_dataset(dataset):
    for column in dataset.columns:
        rows_to_remove = dataset[dataset[column].isna()]
        dataset = dataset.drop(rows_to_remove.index)
    return dataset.drop_duplicates()

def normalise(subset, category_columns, columns_to_ignore):
    for column in subset.columns:
        if not column in category_columns and (column not in columns_to_ignore):
            subset[column] = zscore(subset[column])

    return subset

def encode_categories_features(subset, category_columns):
    subset_category_columns = []
    for column in subset.columns:
        if column in category_columns:
            subset[column] = subset[column].astype('category')
            subset[column] = subset[column].cat.codes
            subset_category_columns.append(column)

    return subset, subset_category_columns

def splitSet(dataset, seed, target_column_name ='target'):
    try:
        training_set, testing_set = train_test_split(
            dataset,
            test_size=0.2,
            random_state=seed,
            stratify=dataset["target"]
        )
    except Exception as e:
        raise  e
    training_set = apply_one_hot_encode(training_set, target_column_name)
    testing_set = apply_one_hot_encode(testing_set, target_column_name)

    feature_columns = [col for col in training_set.columns if not col.startswith(target_column_name + "_")]
    target_columns  = [col for col in training_set.columns if col.startswith(target_column_name + "_")]
    try:
        return (training_set[feature_columns], training_set[target_columns]), (testing_set[feature_columns], testing_set[target_columns])
    except KeyError as e:
        raise KeyError(f"KeyError during train-test split: {e}. Check if the target column exists in the dataset.") from e

def remap_targets(dataset, target_column_name ='target'):
    unique_values = sorted(dataset[target_column_name].unique())
    mapping = {old: new for new, old in enumerate(unique_values, start=1)}
    dataset[target_column_name] = dataset[target_column_name].map(mapping)
    return dataset

def apply_one_hot_encode(dataset, target_column_name ='target'):
    encoded_columns = pd.get_dummies(dataset[target_column_name], prefix=target_column_name)
    dataset = dataset.drop(columns=[target_column_name])
    dataset = pd.concat([dataset, encoded_columns], axis=1)
    return dataset

def prepared_meta_feature_dataset(dataset, target_columns, target_column, need_split=True):
    seed = random.randint(0, 4294967295)

    dataset_x = np.array(dataset.drop(target_columns, axis=1))
    dataset_y = apply_one_hot_encode(dataset, target_column)
    dataset_y = dataset_y[[col for col in dataset_y.columns if col.startswith(f"{target_column}_")]]
    if need_split:
        training_x, validation_x, training_y, validation_y = train_test_split(
            dataset_x,
            dataset_y,
            test_size=0.2,
            random_state=seed,
            stratify=dataset_y
        )

        return (training_x, training_y), (validation_x, validation_y)
    else:
        return dataset_x, dataset_y