import numpy as np
from sklearn.feature_selection import mutual_info_classif


def calculate_meta_features(dataset, category_columns):
    meta_features = {}
    category_columns = list(set(dataset.columns.tolist()) & set(category_columns))

    #Calculate basic meta features
    meta_features["number_of_features"] = dataset.shape[1] - 1
    #what if here are no category fields change to %
    meta_features["proportion_of_numeric_features"] = (meta_features["number_of_features"] - len(category_columns)) / meta_features["number_of_features"]
    meta_features["number_of_instances"] = dataset.shape[0]
    meta_features["number_of_classes"] = len(dataset['target'].unique().tolist())
    meta_features["ratio_of_instances_to_features"] = meta_features["number_of_instances"]/meta_features["number_of_features"]
    meta_features["ratio_of_classes_to_features"] = meta_features["number_of_classes"]/meta_features["number_of_features"]
    meta_features["ratio_of_instances_to_classes"] = meta_features["number_of_instances"]/meta_features["number_of_classes"]
    instance_per_class = dataset['target'].value_counts()
    meta_features["ratio_of_min_to_max_instances_per_class"] = instance_per_class.min()/instance_per_class.max()
    meta_features["proportion_of_features_with_outliers"] = count_number_of_features_with_outliers(dataset, category_columns) / meta_features["number_of_features"]

    #Calculate information meta features
    mi_scores = calculate_mutual_information(dataset, category_columns)
    meta_features["average_mutual_information"] = np.mean(mi_scores)
    meta_features["minimum_mutual_information"] = np.min(mi_scores)
    meta_features["maximum_mutual_information"] = np.max(mi_scores)
    meta_features["equivalent_number_of_features"] = np.exp(np.mean(mi_scores))
    meta_features["noise_to_signal_ratio_of_features"] = calculate_nsr(mi_scores, dataset)

    return meta_features

def calculate_mutual_information(dataset, category_columns):
    y = dataset["target"]
    x = dataset.drop(columns=["target"])

    discrete_mask = [col in category_columns for col in x.columns]

    return mutual_info_classif(x, y, discrete_features=discrete_mask, random_state=42)

def calculate_nsr(mi_scores, dataset):
    y = dataset["target"]

    # Total Mutual Information (I)
    total_mi = np.sum(mi_scores)

    # Calculate Entropy (H) of the target
    p_y = y.value_counts(normalize=True)
    entropy = -np.sum(p_y * np.log(p_y))

    if total_mi == 0:
        return float('inf')

    nsr = (entropy - total_mi) / total_mi
    return nsr

def count_number_of_features_with_outliers(dataset, category_columns):
    number_of_features_without_outliers = 0
    for column in dataset.columns:
        if not column in category_columns and column != "target":
            q1 = dataset[column].quantile(0.25)  # First quartile
            q3 = dataset[column].quantile(0.75)  # Third quartile
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if ((dataset[column] < lower_bound) | (dataset[column] > upper_bound)).any():
                number_of_features_without_outliers += 1
            else:
                mean = dataset[column].mean()
                std_dev = dataset[column].std()

                # Define thresholds
                lower_threshold = mean - 3 * std_dev
                upper_threshold = mean + 3 * std_dev
                if ((dataset[column] < lower_threshold) | (dataset[column] > upper_threshold)).any():
                    number_of_features_without_outliers += 1
    return number_of_features_without_outliers