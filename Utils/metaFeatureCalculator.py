
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from Utils.datasetHandler import loadRawDataset


def calculateMetaFeatures(dataset, categoryColumns):
    metaFeatures = {}
    categoryColumns = list(set( dataset.columns.tolist()) & set(categoryColumns))

    #Calculate basic meta features
    metaFeatures["number_of_features"] = dataset.shape[1] - 1
    #what if here are no category fields change to %
    metaFeatures["proportion_of_numeric_features"] =  (metaFeatures["number_of_features"] - len(categoryColumns))/metaFeatures["number_of_features"]
    metaFeatures["number_of_instances"] = dataset.shape[0]
    metaFeatures["number_of_classes"] = len(dataset['target'].unique().tolist())
    metaFeatures["ratio_of_instances_to_features"] = metaFeatures["number_of_instances"]/metaFeatures["number_of_features"]
    metaFeatures["ratio_of_classes_to_features"] = metaFeatures["number_of_classes"]/metaFeatures["number_of_features"]
    metaFeatures["ratio_of_instances_to_classes"] = metaFeatures["number_of_instances"]/metaFeatures["number_of_classes"]
    instancePerClass = dataset['target'].value_counts()
    metaFeatures["ratio_of_min_to_max_instances_per_class"] = instancePerClass.min()/instancePerClass.max()
    metaFeatures["proportion_of_features_with_outliers"] = countNumberOfFeaturesWithOutliers(dataset, categoryColumns)/metaFeatures["number_of_features"]

    #Calculate information meta features
    miScores = calculateMutualInformation(dataset, categoryColumns)
    metaFeatures["average_mutual_information"] = calculateAverageMiScore(miScores)
    metaFeatures["equivalent_number_of_features"] = calculateEna(miScores[0], miScores[1])
    metaFeatures["noise_to_signal_ratio_of_features"] = calculateNsr(miScores[0], miScores[1], dataset)

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

    return [normaliseMiScores(miContinuous), normaliseMiScores(miDiscrete)]

def calculateAverageMiScore(miScores):
    return miScores.average()

def calculateEna(miContinuous, miDiscrete):
    # Calculate Equivalent Number of Features
    miScores = np.concatenate((miContinuous, miDiscrete))
    return np.exp(np.mean(miScores))

def calculateNsr(miContinuous, miDiscrete, dataset):
    Y = dataset["target"]

    # Total Mutual Information (I)
    totalMi = np.sum(miContinuous) + np.sum(miDiscrete)

    # Calculate Entropy (H) of the target
    pY = Y.value_counts(normalize=True)  # Probability distribution of target values
    entropy = -np.sum(pY * np.log2(pY))

    if totalMi == 0:
        return float('inf')

    # Calculate Noise-Signal Ratio (NSR)
    return (entropy - totalMi) / totalMi

def countNumberOfFeaturesWithOutliers(dataset, categoryColumns):
    numberOfFeaturesWithoutOutliers = 0
    for column in dataset.columns:
        if not column in categoryColumns and column != "target":
            Q1 = dataset[column].quantile(0.25)  # First quartile
            Q3 = dataset[column].quantile(0.75)  # Third quartile
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if ((dataset[column] < lower_bound) | (dataset[column] > upper_bound)).any():
                numberOfFeaturesWithoutOutliers += 1
            else:
                mean = dataset[column].mean()
                std_dev = dataset[column].std()

                # Define thresholds
                lower_threshold = mean - 3 * std_dev
                upper_threshold = mean + 3 * std_dev
                if ((dataset[column] < lower_threshold) | (dataset[column] > upper_threshold)).any():
                    numberOfFeaturesWithoutOutliers += 1
    return numberOfFeaturesWithoutOutliers

def normaliseMiScores(miScores):
    if len(miScores) == 0 or np.all(miScores == 0):
        return miScores
    return (miScores - miScores.min()) / (miScores.max() - miScores.min())