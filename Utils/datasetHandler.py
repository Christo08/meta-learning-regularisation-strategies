import math
import random
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from pmlb import fetch_data
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

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

def create_subsets_with_seeds(databaseName, numberOfSubsetsNeed, classSeeds, featuresSeeds, instancesSeeds, datasetSettings):
    print("Recreating "+str(numberOfSubsetsNeed)+" Subsets for the "+databaseName+" dataset")
    dataset = loadRawDataset(datasetSettings)
    dataset = cleanDataset(dataset)
    numeric_data = dataset.select_dtypes(include=[np.number])
    assert not np.isinf(numeric_data.values).any(), "Inf in numeric input DataFrame"

    classSubsets, classSeeds = makeClassesSubsets(dataset, numberOfSubsetsNeed, classSeeds)

    featuresSubsets, featuresSeeds = makeFeaturesSubsets(dataset, len(featuresSeeds), featuresSeeds)

    instancesSubsets, instancesSeeds = makeInstancesSubsets(dataset, len(instancesSeeds), instancesSeeds)

    subsets = classSubsets + featuresSubsets + instancesSubsets
    seeds = classSeeds + featuresSeeds + instancesSeeds

    subsets = np.array(subsets, dtype=object)

    returnSubsets = []
    metaFeatures = []
    subsetsCategoryColumns = []
    for subset, seed in zip(subsets, seeds):
        subset, subsetCategoryColumns = encodeCategoriesFeatures(subset, datasetSettings['categoryColumns'])
        subset = remapTargets(subset)

        metaFeatures.append(calculateMetaFeatures(subset, datasetSettings['categoryColumns']))

        subset = normalise(subset, subsetCategoryColumns,["target"])

        subset.reset_index(drop=True, inplace=True)
        returnSubsets.append(subset)
        subsetsCategoryColumns.append(subsetCategoryColumns)

    return returnSubsets, metaFeatures, seeds, subsetsCategoryColumns

def load_subset(filePath, seed, datasetSettings):
    subset = pd.read_csv(filePath)
    trainingSet, testingSet = splitSet(subset, seed)
    fullCategoryColumns = datasetSettings['categoryColumns']
    categoryColumns = []
    for categoryColumn in fullCategoryColumns:
        if categoryColumn in subset.columns:
            categoryColumns.append(categoryColumn)

    return trainingSet, testingSet, categoryColumns

def create_subsets(databaseName, numberOfSubsetsNeed, datasetSettings, needSplit=True):
    print("Creating "+str(numberOfSubsetsNeed)+" Subsets for the "+databaseName+" dataset")
    dataset = loadRawDataset(datasetSettings)
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

    returnSubsets = []
    trainingSets = []
    testingSets = []
    metaFeatures = []
    subsetsCategoryColumns = []
    for subset, seed in zip(subsets, seeds):
        subset, subsetCategoryColumns = encodeCategoriesFeatures(subset, datasetSettings['categoryColumns'])
        subset = remapTargets(subset)

        metaFeatures.append(calculateMetaFeatures(subset, datasetSettings['categoryColumns']))

        subset = normalise(subset, subsetCategoryColumns,["target"])

        subset.reset_index(drop=True, inplace=True)
        if needSplit:
            trainingSet, testingSet = splitSet(subset, seed["seed"])

            trainingSets.append(trainingSet)
            testingSets.append(testingSet)
        else:
            returnSubsets.append(subset)
        subsetsCategoryColumns.append(subsetCategoryColumns)

    if needSplit:
        return trainingSets, testingSets, metaFeatures, seeds, subsetsCategoryColumns
    else:
        return returnSubsets, metaFeatures, seeds, subsetsCategoryColumns

def load_dataset(datasetSettings):
    dataset = loadRawDataset(datasetSettings)
    dataset = cleanDataset(dataset)
    numeric_data = dataset.select_dtypes(include=[np.number])
    assert not np.isinf(numeric_data.values).any(), "Inf in numeric input DataFrame"

    seed = random.randint(1, 100000)
    random.seed(seed)

    dataset, datasetCategoryColumns = encodeCategoriesFeatures(dataset, datasetSettings['categoryColumns'])
    metaFeatures = calculateMetaFeatures(dataset, datasetSettings['categoryColumns'])

    dataset = normalise(dataset, datasetCategoryColumns, ["target"])

    dataset = remapTargets(dataset)

    dataset.reset_index(drop=True, inplace=True)

    trainingSet, testingSet = splitSet(dataset, seed)

    return [trainingSet], [testingSet], [metaFeatures], [seed], [datasetCategoryColumns]

def loadOptimiserDataset(seed, datasetSettings):
    dataset = loadRawDataset(datasetSettings)
    dataset = cleanDataset(dataset)

    target_columns = [col for col in dataset.columns if 'target' in col]
    dataset = normalise(dataset, datasetSettings['categoryColumns'], target_columns)

    return splitSet(dataset, seed), datasetSettings['categoryColumns']

def applySMOTE(x, y, seed, numberOfNeighbors, categoryColumns):
    if isinstance(y, pd.DataFrame):
        y = y.idxmax(axis=1).str.extract(r'(\d+)$').astype(int).squeeze()

    classCounts = Counter(y.values.ravel())
    minClassSamples = min(classCounts.values())
    if minClassSamples <= 1:
        raise ValueError("Cannot apply smote.")

    safeNeighbors = min(numberOfNeighbors, minClassSamples - 1)
    if safeNeighbors < 1:
        raise ValueError("Cannot apply smote.")

    if not categoryColumns:
        oversample = SMOTE(random_state=seed, k_neighbors=safeNeighbors)
    else:
        categoryIndices = [x.columns.get_loc(col) for col in categoryColumns]
        oversample = SMOTENC(random_state=seed, categorical_features=categoryIndices, k_neighbors=safeNeighbors)

    assert x.shape[0] == y.shape[0], "Mismatched number of samples between X and Y before applying SMOTE."

    xResampled, yResampled = oversample.fit_resample(x, y)
    assert x.shape[0] == y.shape[0], "Mismatched number of samples between X and Y after applying SMOTE."

    encodedY = pd.get_dummies(yResampled).astype(int)

    return xResampled, encodedY

#helper function
def makeClassesSubsets(dataset, numberOfSubsetsNeed, seeds=None):
    if seeds is None:
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
    newSeeds =  len(seeds) < numberOfClassSubset
    while counter < numberOfUniqueClassesCombos and len(subsets) < numberOfClassSubset:
        if newSeeds:
            seed = random.randint(1, 100000)
        else:
            seed = seeds[counter]["seed"]

        comboSize = random.randint(1, numberOfClasses - MIN_CLASSES_REQUIRED)
        combo = tuple(sorted(random.sample(classLabels, comboSize)))

        if combo in usedClassCombos:
            continue

        usedClassCombos.add(combo)
        subset = dataset[~dataset['target'].isin(combo)]

        if len(subset) < MIN_INSTANCES_PER_SUBSET:
            continue

        subsets.append(subset.copy())
        if newSeeds:
            seeds.append({"seed": seed, "subsetType": "classes"})

        counter += 1

    return subsets, seeds

def makeFeaturesSubsets(dataset, numberOfFeatureSubset, seeds=None):
    if seeds is None:
        seeds = []
    subsets = []
    features = [col for col in dataset.columns if col != 'target']

    numberOfFeatures = len(features)
    maximumSubsetSize = round(numberOfFeatures * MIN_FEATURE_FRACTION)

    counter = 0
    numberOfUniqueFeatureCombos = sum(math.comb(numberOfFeatures, k) for k in range(1, maximumSubsetSize)) - 1

    usedFeaturesCombos = set()
    newSeeds =  len(seeds) < numberOfFeatureSubset
    while counter < numberOfUniqueFeatureCombos and len(subsets) < numberOfFeatureSubset:
        if newSeeds:
            seed = random.randint(1, 100000)
        else:
            seed = seeds[counter]["seed"]
        random.seed(seed)

        comboSize = random.randint(1, maximumSubsetSize)
        combo = tuple(sorted(random.sample(features, comboSize)))

        if combo in usedFeaturesCombos:
            continue

        subset = dataset.drop(columns=list(combo))

        subsets.append(subset.copy())
        if newSeeds:
            seeds.append({"seed": seed, "subsetType": "features"})

        counter += 1

    return subsets,seeds

def makeInstancesSubsets(dataset, numberOfInstancesSubsetsNeeded, seeds=None):
    if seeds is None:
        seeds = []
    subsets = []
    numberOfInstances = len(dataset)
    intervalSize = (numberOfInstances - MIN_INSTANCES_PER_SUBSET) // numberOfInstancesSubsetsNeeded
    newSeeds =  len(seeds) < numberOfInstancesSubsetsNeeded

    for counter in range(numberOfInstancesSubsetsNeeded):
        if newSeeds:
            seed = random.randint(1, 100000)
        else:
            seed = seeds[counter]["seed"]

        offset = random.randint(OFFSET_RANGE_START, intervalSize)
        subsetSize = MIN_INSTANCES_PER_SUBSET + intervalSize * counter + offset
        subsetSize = min(subsetSize, math.floor(numberOfInstances-dataset.shape[0]*0.1))
        createSubset = True
        while createSubset:
            subset, _ = train_test_split(
                dataset,
                train_size=subsetSize,
                stratify=dataset['target'],
                random_state=seed
            )
            createSubset = False
            instancePerClasses = subset['target'].value_counts()
            for target, count in instancePerClasses.items():
                if count < 10:
                    if len(subset['target'].unique()) > MIN_CLASSES_REQUIRED:
                        subset = subset[~subset['target'].isin([target])]
                    else:
                        createSubset = True
                        break


        subsets.append(subset.copy())
        if newSeeds:
            seeds.append({"seed": seed, "subsetType": "instances"})

    return subsets, seeds

def loadRawDataset(datasetSettings):
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
    return dataset

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

def splitSet(dataset, seed, targetColumnName = 'target'):
    try:
        trainingSet, testingSet = train_test_split(
            dataset,
            test_size=0.2,
            random_state=seed,
            stratify=dataset["target"]
        )
    except Exception as e:
        raise  e
    trainingSet = applyOneHotEncode(trainingSet, targetColumnName)
    testingSet = applyOneHotEncode(testingSet, targetColumnName)

    featureColumns = [col for col in trainingSet.columns if not col.startswith(targetColumnName + "_")]
    targetColumns  = [col for col in trainingSet.columns if col.startswith(targetColumnName + "_")]
    try:
        return (trainingSet[featureColumns], trainingSet[targetColumns]), (testingSet[featureColumns], testingSet[targetColumns])
    except KeyError as e:
        raise KeyError(f"KeyError during train-test split: {e}. Check if the target column exists in the dataset.") from e

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