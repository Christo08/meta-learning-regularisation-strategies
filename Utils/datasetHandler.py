import math
import random
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, SMOTENC
from pmlb import fetch_data
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

from Utils.fileHandler import loadDatasetSetting
from Utils.fileHandler import loadMetaFeaturesCSV
from Utils.metaFeatureCalculator import calculateMetaFeatures

dataset = pd.DataFrame()
datasetName = ""
categoryColumns = []

# Constants
MIN_CLASSES_REQUIRED = 2
MIN_INSTANCES_PER_SUBSET = 100
MIN_FEATURE_FRACTION = 0.5
OFFSET_RANGE_START = 1

def loadMetaFeaturesDataset(seed, isSVM):
    dataset = loadMetaFeaturesCSV()

    columns_to_drop = [
        "batch_size", "dropout_layers", "learning_rate", "momentum", "number_of_epochs", "number_of_hidden_layers",
        "number_of_neurons_in_layers", "prune_amount", "prune_epoch_interval", "weight_decay",
        "weight_perturbation_amount", "weight_perturbation_interval", "dataset_name", "seed", "normal_training_loss",
        "normal_validation_loss", "batch_normalisation_training_loss", "batch_normalisation_validation_loss",
        "dropout_training_loss", "dropout_validation_loss", "layer_normalisation_training_loss",
        "layer_normalisation_validation_loss", "SMOTE_training_loss", "SMOTE_validation_loss", "prune_training_loss",
        "prune_validation_loss", "weight_decay_training_loss", "weight_decay_validation_loss",
        "weight_normalisation_training_loss", "weight_normalisation_validation_loss",
        "weight_perturbation_training_loss", "weight_perturbation_validation_loss", "best_training_technique",
        "best_validation_technique", "best_testing_technique"
    ]
    targetColumns = ["normal_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                      "layer_normalisation_testing_loss", "SMOTE_testing_loss", "prune_testing_loss",
                      "weight_decay_testing_loss", "weight_normalisation_testing_loss", "weight_perturbation_testing_loss"]

    dataset.drop(columns=columns_to_drop, errors="ignore", inplace=True)

    for column in dataset.columns:
        columnValues = dataset[column].replace([np.inf, -np.inf], np.nan)
        maxFinite = columnValues.max(skipna=True)

        if pd.isna(maxFinite):
            maxFinite = 1e6

        dataset[column] = dataset[column].replace([np.inf, -np.inf], maxFinite)

    ranks = dataset[targetColumns].rank(axis=1, method="dense", ascending=True)
    dataset.drop(columns=targetColumns, errors="ignore", inplace=True)

    dataset_with_ranks = pd.concat([dataset, ranks], axis=1)
    dataset_with_ranks = normalise(dataset_with_ranks, targetColumns)

    subset, testingSet = train_test_split(
        dataset_with_ranks, test_size=0.2, random_state=seed
    )
    testingSetX = testingSet.drop(columns=targetColumns)
    subsetX = subset.drop(columns=targetColumns)
    testingSetY = testingSet[targetColumns]
    subsetY = subset[targetColumns]

    return (subsetX, subsetY), (testingSetX, testingSetY)

def createSubsets(databaseName, numberOfSubsetsNeed):
    dataset, datasetSettings = loadDataset(databaseName)
    dataset = cleanDataset(dataset)
    numeric_data = dataset.select_dtypes(include=[np.number])
    assert not np.isinf(numeric_data.values).any(), "Inf in numeric input DataFrame"

    classSubsets, classSeeds = makeClassesSubsets(dataset, numberOfSubsetsNeed)
    numberOfFeatureSubset = (numberOfSubsetsNeed - len(classSubsets)) //2

    featuresSubsets, featuresSeeds = makeFeaturesSubsets(dataset, numberOfFeatureSubset)
    numberOfInstanceSubset = numberOfSubsetsNeed - len(featuresSubsets) - len(classSubsets)

    instancesSubsets, instancesSeeds = makeInstancesSubsets(dataset, numberOfInstanceSubset)

    subsets = classSubsets + featuresSubsets + instancesSubsets
    seeds = classSeeds + featuresSeeds + instancesSeeds

    subsets = np.array(subsets, dtype=object)

    trainingSets = []
    testingSets = []
    metaFeatures = []
    subsetsCategoryColumns = []
    for subset, seed in zip(subsets, seeds):
        subset, subsetCategoryColumns = encodeCategoriesFeatures(subset, datasetSettings['categoryColumns'])

        metaFeatures.append(calculateMetaFeatures(subset, datasetSettings['categoryColumns']))
        subset = normalise(subset, subsetCategoryColumns,["target"])

        subset = remapTargets(subset)
        subset = applyOneHotEncode(subset)

        subset.reset_index(drop=True, inplace=True)

        trainingSet, testingSet = splitSet(subset, seed)
        trainingSets.append(trainingSet)
        testingSets.append(testingSet)
        subsetsCategoryColumns.append(subsetCategoryColumns)


    return trainingSets, testingSets, metaFeatures, seeds, subsetsCategoryColumns

def loadOptimiserDataset(databaseName, seed):
    dataset, datasetSettings = loadDataset(databaseName)
    dataset = cleanDataset(dataset)
    dataset = applyOneHotEncode(dataset)

    return splitSet(dataset, seed), datasetSettings['categoryColumns']

def applySMOTE(x, y, seed, numberOfNeighbors, categoryColumns):

    # Convert one-hot encoded labels (DataFrame) to class indices (Series)
    if isinstance(y, pd.DataFrame):
        y = y.idxmax(axis=1).str.extract(r'(\d+)$').astype(int).squeeze()

    # Check class distribution
    classCounts = Counter(y.values.ravel())
    minClassSamples = min(classCounts.values())
    if minClassSamples <= 1:
        raise ValueError("Cannot apply smote.")

    # Adjust neighbors to be valid
    safeNeighbors = min(numberOfNeighbors, minClassSamples - 1)
    if safeNeighbors < 1:
        raise ValueError("Cannot apply smote.")

    # Choose SMOTE or SMOTENC
    if not categoryColumns:
        oversample = SMOTE(random_state=seed, k_neighbors=safeNeighbors)
    else:
        categoryIndices = [x.columns.get_loc(col) for col in categoryColumns]
        oversample = SMOTENC(random_state=seed, categorical_features=categoryIndices, k_neighbors=safeNeighbors)

    assert x.shape[0] == y.shape[0], "Mismatched number of samples between X and Y before applying SMOTE."
    # Apply oversampling
    xResampled, yResampled = oversample.fit_resample(x, y)
    assert x.shape[0] == y.shape[0], "Mismatched number of samples between X and Y after applying SMOTE."

    # Convert y back to one-hot encoded DataFrame
    encodedY = pd.get_dummies(yResampled).astype(int)

    return xResampled, encodedY

#helper function
def makeClassesSubsets(dataset, numberOfSubsetsNeed):
    seeds = []
    subsets = []
    classLabels = dataset['target'].unique().tolist()
    numberOfClasses = len(classLabels)
    if numberOfClasses <= MIN_CLASSES_REQUIRED:
        return subsets, seeds

    numberOfClassSubset = numberOfSubsetsNeed // 3
    numberOfUniqueClassesCombos = sum(math.comb(numberOfClasses, k) for k in range(1, numberOfClasses - MIN_CLASSES_REQUIRED)) - 1
    counter = 0
    usedClassCombos = set()
    while counter < numberOfUniqueClassesCombos and len(subsets) < numberOfClassSubset:
        seed = random.randint(1, 100000)
        random.seed(seed)

        comboSize = random.randint(1, numberOfClasses - MIN_CLASSES_REQUIRED)
        combo = tuple(sorted(random.sample(classLabels, comboSize)))

        if combo in usedClassCombos:
            continue

        usedClassCombos.add(combo)
        subset = dataset[~dataset['class'].isin(combo)]

        if len(subset) < MIN_INSTANCES_PER_SUBSET:
            continue

        subsets.append(subset.copy())
        seeds.append(seed)

        counter += 1

    return subsets, seeds

def makeFeaturesSubsets(dataset, numberOfFeatureSubset):
    seeds = []
    subsets = []
    features = [col for col in dataset.columns if col != 'target']

    numberOfFeatures = len(features)
    maximumSubsetSize = round(numberOfFeatures * MIN_FEATURE_FRACTION)

    counter = 0
    numberOfUniqueFeatureCombos = sum(math.comb(numberOfFeatures, k) for k in range(1, maximumSubsetSize)) - 1

    usedFeaturesCombos = set()
    while counter < numberOfUniqueFeatureCombos and len(subsets) < numberOfFeatureSubset:
        seed = random.randint(1, 100000)
        random.seed(seed)

        comboSize = random.randint(1, maximumSubsetSize)
        combo = tuple(sorted(random.sample(features, comboSize)))

        if combo in usedFeaturesCombos:
            continue

        subset = dataset.drop(columns=list(combo))

        subsets.append(subset.copy())
        seeds.append(seed)

        counter += 1

    return subsets,seeds

def makeInstancesSubsets(dataset, numberOfInstancesSubsetsNeeded):
    seeds = []
    subsets = []
    numberOfInstances = len(dataset)
    intervalSize = (numberOfInstances - MIN_INSTANCES_PER_SUBSET) // numberOfInstancesSubsetsNeeded

    for counter in range(numberOfInstancesSubsetsNeeded):
        seed = random.randint(1, 100000)
        random.seed(seed)

        offset = random.randint(OFFSET_RANGE_START, intervalSize)
        subsetSize = MIN_INSTANCES_PER_SUBSET + intervalSize * counter + offset
        subsetSize = min(subsetSize, numberOfInstances)

        subset, _ = train_test_split(
            dataset,
            train_size=subsetSize,
            stratify=dataset['target'],
            random_state=seed
        )

        subsets.append(subset.copy())
        seeds.append(seed)

    return subsets, seeds

def loadDataset(databaseName):
    datasetsSettings = loadDatasetSetting()
    datasetSettings = None
    for dataset in datasetsSettings:
        if databaseName == dataset['name']:
            datasetSettings = dataset
            break
    if not datasetSettings:
        raise Exception("Dataset not fond")

    if datasetSettings["type"] == "csv":
        dataset = pd.read_csv(datasetSettings["filePath"])
    else:
        dataset = fetch_data(datasetSettings["pmlbName"])

    if not(datasetSettings["targetColumn"] == "target"):
        dataset.rename(columns={datasetSettings["targetColumn"]: "target"}, inplace=True)

    if not(len(datasetSettings["dropColumns"]) == 0):
        dataset = dataset.drop(columns=datasetSettings["dropColumns"])

    dataset['target'] = dataset['target'].astype('category')
    dataset['target'] = dataset['target'].cat.codes
    return dataset,datasetSettings

def cleanDataset(dataset):
    for column in dataset.columns:
        rowsToRemove = dataset[dataset[column].isna()]
        dataset = dataset.drop(rowsToRemove.index)
    return dataset.drop_duplicates()

def normalise(subset, categoryColumns, columnsToIgnore):
    for column in subset.columns:
        if not column in categoryColumns and (column not in columnsToIgnore):
            subset[column] = zscore(subset[column])

    return subset

def encodeCategoriesFeatures(subset, categoryColumns):
    subsetCategoryColumns = []
    for column in subset.columns:
        if column in categoryColumns:
            subset[column] = subset[column].astype('category')
            subset[column] = subset[column].cat.codes
            subsetCategoryColumns.append(column)

    return subset, subsetCategoryColumns

def splitSet(dataset, seed):
    dataset, testingSet = train_test_split(
        dataset, test_size=0.2, random_state=seed
    )
    columnNames = dataset.columns.tolist()
    featureColumnNames = []
    targetColumnNames = []
    for column in columnNames:
        if 'target' in column:
            targetColumnNames.append(column)
        else:
            featureColumnNames.append(column)

    datasetY = dataset[targetColumnNames]
    datasetX = dataset[featureColumnNames]

    testingSetY = testingSet[targetColumnNames]
    testingSetX = testingSet[featureColumnNames]

    return (datasetX, datasetY), (testingSetX, testingSetY)

def remapTargets(dataset, targetColumnName = 'target'):
    uniqueValues = sorted(dataset[targetColumnName].unique())
    mapping = {old: new for new, old in enumerate(uniqueValues, start=1)}
    dataset[targetColumnName] = dataset[targetColumnName].map(mapping)
    return dataset

def applyOneHotEncode(dataset, targetColumnName = 'target'):
    encodedColumns = pd.get_dummies(dataset[targetColumnName], prefix=targetColumnName)
    dataset = dataset.drop(columns=[targetColumnName])
    dataset = pd.concat([dataset, encodedColumns], axis=1)
    return dataset

def cleanLabels(targets, num_classes, is_subset = True):
    # Convert pandas Series to numpy array if necessary
    if isinstance(targets, pd.Series):
        targets = targets.to_numpy()

    # Convert to torch tensor if not already a tensor
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.int64)

    # Validate that all target values exist in the mapping
    mapped_targets = []
    if not is_subset:
        unique_classes = sorted(targets.unique().tolist())
        newClassMapping = {oldClass: newClass for newClass, oldClass in enumerate(unique_classes)}
    for target in targets:
        if target.item() not in newClassMapping:
            raise ValueError(f"Value {target.item()} in targets is not in the class mapping: {newClassMapping}")
        mapped_targets.append(newClassMapping[target.item()])

    targets = torch.tensor(mapped_targets, dtype=torch.int64)

    # Initialize output tensor
    num_samples = targets.size(0)
    output_tensor = torch.zeros(num_samples, num_classes)

    # Populate the one-hot encoding
    output_tensor.scatter_(1, targets.unsqueeze(1), 1)
    output_tensor = output_tensor.to(torch.float)

    return output_tensor

def getClassCombinations(dataset, numberOfClasses):
    global minimumNumberOfClasses, minimumPercentsOfInstances

    if numberOfClasses <= minimumNumberOfClasses:
        return []

    classCombos = []
    for r in range(1, numberOfClasses - minimumNumberOfClasses + 1):
        classCombos.extend([set(c) for c in combinations(range(numberOfClasses), r)])

    counts = Counter(dataset["target"])
    maximumInstanceDroppingAllowed = dataset.shape[0] * (1 - minimumPercentsOfInstances)
    for counter in reversed(range(len(classCombos))):
        totalDropping = 0
        classCombo = classCombos[counter]
        for clazz in classCombo:
            totalDropping += counts[clazz]
            if totalDropping > maximumInstanceDroppingAllowed:
                classCombos.remove(classCombo)
                break

    return classCombos

def getFeaturesCombinations(numberFeatures):
    global minimumPercentsOfFeatures
    minimumNumberOfFeatures = math.floor(numberFeatures * minimumPercentsOfFeatures)

    featuresCombos = []
    for r in range(1, numberFeatures - minimumNumberOfFeatures + 1):
        featuresCombos.extend([set(c) for c in combinations(range(minimumNumberOfFeatures), r)])

    return featuresCombos