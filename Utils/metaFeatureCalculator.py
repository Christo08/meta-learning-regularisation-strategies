
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def calculateMetaFeatures(dataset, categoryColumns):
    metaFeatures = {}
    categoryColumns = list(set( dataset.columns.tolist()) & set(categoryColumns))

    #Calculate basic meta features
    metaFeatures["number_of_attributes"] = dataset.shape[1] - 1
    #what if here are no category fields change to %
    metaFeatures["percentage_of_numeric_features"] =  (metaFeatures["number_of_attributes"] - len(categoryColumns))/metaFeatures["number_of_attributes"]
    metaFeatures["number_of_instances"] = dataset.shape[0]
    metaFeatures["number_of_class"] = len(dataset['target'].unique().tolist())
    metaFeatures["proportion_of_attributes_per_instances"] = metaFeatures["number_of_attributes"]/metaFeatures["number_of_instances"]
    metaFeatures["number_of_classes_per_attributes"] = metaFeatures["number_of_class"]/metaFeatures["number_of_attributes"]
    metaFeatures["number_of_instances_per_class"] = metaFeatures["number_of_instances"]/metaFeatures["number_of_class"]
    instancePerClass = dataset['target'].value_counts()
    metaFeatures["proportion_of_min_to_max_instance_per_class"] = instancePerClass.min()/instancePerClass.max()
    metaFeatures["proportion_of_attributes_with_outliers"] = countNumberOfAttributesWithOutliers(dataset, categoryColumns)/metaFeatures["number_of_attributes"]

    #Calculate information meta features
    miScores = calculateMutualInformation(dataset, categoryColumns)
    metaFeatures["minimum_continuous_mutual_information"],metaFeatures["maximum_continuous_mutual_information"] = calculateMinAndMaxMiScores(miScores[0])
    metaFeatures["minimum_discrete_mutual_information"],metaFeatures["maximum_discrete_mutual_information"] = calculateMinAndMaxMiScores(miScores[1])
    metaFeatures["equivalent_number_of_continuous_attributes"] = calculateEna(miScores[0])
    metaFeatures["equivalent_number_of_discrete_attributes"] = calculateEna(miScores[1])
    metaFeatures["noise_signal_ratio_of_continuous_attributes"] = calculateNsr(miScores[0], dataset)
    metaFeatures["noise_signal_ratio_of_discrete_attributes"] = calculateNsr(miScores[1], dataset)

    return metaFeatures

def calculateMutualInformation(dataset, categoryColumns):
    X = dataset.drop(columns=["target"])
    Y = dataset["target"]

    XDiscrete = X[categoryColumns]
    XContinuous = X.drop(columns=categoryColumns)

    # Continuous mutual information
    if XContinuous.shape[1] > 0:
        miContinuous = mutual_info_regression(XContinuous, Y)
    else:
        miContinuous = np.array([0])

    # Discrete mutual information
    if XDiscrete.shape[1] > 0:
        miDiscrete = mutual_info_classif(XDiscrete, Y, discrete_features=True)
    else:
        miDiscrete = np.array([0])

    return [miContinuous, miDiscrete]

def calculateMinAndMaxMiScores(miScores):
    return miScores.min(), miScores.max()

def calculateEna(miScores):
    # Calculate Equivalent Number of Attributes
    return np.exp(np.mean(miScores))

def calculateNsr(miScores, dataset):
    Y = dataset["target"]

    # Total Mutual Information (I)
    totalMi = np.sum(miScores)

    # Calculate Entropy (H) of the target
    pY = Y.value_counts(normalize=True)  # Probability distribution of target values
    entropy = -np.sum(pY * np.log2(pY))

    if totalMi == 0:
        return float('inf')

    # Calculate Noise-Signal Ratio (NSR)
    return (entropy - totalMi) / totalMi

def countNumberOfAttributesWithOutliers(dataset, categoryColumns):
    numberOfAttributesWithoutOutliers = 0
    for column in dataset.columns:
        if not column in categoryColumns and column != "target":
            Q1 = dataset[column].quantile(0.25)  # First quartile
            Q3 = dataset[column].quantile(0.75)  # Third quartile
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if ((dataset[column] < lower_bound) | (dataset[column] > upper_bound)).any():
                numberOfAttributesWithoutOutliers += 1
            else:
                mean = dataset[column].mean()
                std_dev = dataset[column].std()

                # Define thresholds
                lower_threshold = mean - 3 * std_dev
                upper_threshold = mean + 3 * std_dev
                if ((dataset[column] < lower_threshold) | (dataset[column] > upper_threshold)).any():
                    numberOfAttributesWithoutOutliers += 1
    return numberOfAttributesWithoutOutliers