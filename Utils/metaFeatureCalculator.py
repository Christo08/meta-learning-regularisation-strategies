import numpy as np
from sklearn.feature_selection import mutual_info_classif


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
    metaFeatures["average_mutual_information"] = np.mean(miScores)
    metaFeatures["minimum_mutual_information"] = np.min(miScores)
    metaFeatures["maximum_mutual_information"] = np.max(miScores)
    metaFeatures["equivalent_number_of_features"] = np.exp(np.mean(miScores))
    metaFeatures["noise_to_signal_ratio_of_features"] = calculateNsr(miScores, dataset)

    return metaFeatures

def calculateMutualInformation(dataset, categoryColumns):
    Y = dataset["target"]
    X = dataset.drop(columns=["target"])

    discreteMask = [col in categoryColumns for col in X.columns]

    miScores = mutual_info_classif(X, Y, discrete_features=discreteMask, random_state=42)
    return miScores

def calculateNsr(miScores, dataset):
    Y = dataset["target"]

    # Total Mutual Information (I)
    totalMi = np.sum(miScores)

    # Calculate Entropy (H) of the target
    pY = Y.value_counts(normalize=True)
    entropy = -np.sum(pY * np.log(pY))

    if totalMi == 0:
        return float('inf')

    nsr = (entropy - totalMi) / totalMi
    return nsr

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

def pairwiseEuclidean(A: np.ndarray) -> np.ndarray:
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    # Efficient squared distance computation
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    s = np.sum(A * A, axis=1)
    D2 = s[:, None] + s[None, :] - 2 * (A @ A.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)


def digammaInteger(n: int) -> float:
    """Digamma(psi) for positive integer n using harmonic numbers.
    psi(n) = H_{n-1} - gamma, where H_m = sum_{i=1}^m 1/i and gamma is Euler-Mascheroni.
    """
    EULER_GAMMA = 0.5772156649015328606
    if n <= 0:
        raise ValueError("digamma integer requires n>=1")
    if n == 1:
     return -EULER_GAMMA
    # compute harmonic number H_{n-1}
    # For moderate n this is fine; vectorization not needed here.
    h = 0.0
    for i in range(1, n):
        h += 1.0 / i
    return h - EULER_GAMMA