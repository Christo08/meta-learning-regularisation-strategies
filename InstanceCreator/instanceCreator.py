import os
import random
import time

import numpy as np
import pandas as pd

from datetime import datetime

from ModelTrainer.nnTrainer import trainNN
from Utils.datasetHandler import createSubsets, loadDataset, createSubsetsWithSeeds, loadSubset
from Utils.fileHandler import loadMetaFeaturesDataset, saveMetaFeaturesDataset, loadSettings
from Utils.timeFormatter import formatDuration

configurations = [
    {"name": "baseline", "param": "baseline", "fileName": "baseline"},
    {"name": "batchNormalisation", "param": "batchNormalisation", "fileName": "batch_normalisation"},
    {"name": "dropout", "param": "dropout", "fileName": "dropout"},
    {"name": "layerNormalisation", "param": "layerNormalisation", "fileName": "layer_normalisation"},
    {"name": "SMOTE", "param": "SMOTE", "fileName": "SMOTE"},
    {"name": "prune", "param": "prune", "fileName": "prune"},
    {"name": "weightDecay", "param": "weightDecay", "fileName": "weight_decay"},
    {"name": "weightNormalisation", "param": "weightNormalisation", "fileName": "weight_normalisation"},
    {"name": "weightPerturbation", "param": "weightPerturbation", "fileName": "weight_perturbation"}
]

def recreateSubsets(metaFeatureDataset, numberOfInstances, datasetsSettings, names=[]):
    seeds = []
    if len(metaFeatureDataset)>0:
        for name, group in metaFeatureDataset.groupby('dataset_name'):
            seed ={
                "name": name,
                "datasetSettings": next((item for item in datasetsSettings if item["name"] == name), None),
                "classSeeds": [],
                "featuresSeeds": [],
                "instancesSeeds": [],
                "isComplete": True
            }
            if group.shape[0] < numberOfInstances:
                seed["isComplete"] = False
            else:
                seed["isComplete"] = True
                for index, row in group.iterrows():
                    if row['subset_type'] == "classes":
                        seed["classSeeds"].append({"seed": row['seed'], "subsetType": row['subset_type']})
                    elif row['subset_type'] == "instances":
                        seed["instancesSeeds"].append({"seed": row['seed'], "subsetType": row['subset_type']})
                    else:
                        seed["featuresSeeds"].append({"seed": row['seed'], "subsetType": row['subset_type']})
            seeds.append(seed)
    else:
        for name in names:
            seed ={
                "name": name,
                "datasetSettings": next((item for item in datasetsSettings if item["name"] == name), None),
                "isComplete": False
            }
            seeds.append(seed)

    metaFeatureDataset = []
    for seed in seeds:
        if seed["isComplete"]:
            subsets, metaFeatures, returnSeeds, subsetCategoryColumns = createSubsetsWithSeeds(seed["name"],
                                                                                               numberOfInstances,
                                                                                               seed["classSeeds"],
                                                                                               seed["featuresSeeds"],
                                                                                               seed["instancesSeeds"],
                                                                                               seed["datasetSettings"])
        else:
            subsets, metaFeatures, returnSeeds, subsetCategoryColumns = createSubsets(seed["name"],
                                                                                      numberOfInstances,
                                                                                      seed["datasetSettings"],
                                                                                      False)
        for subset, metaFeature, returnSeed, categoryColumns in zip(subsets, metaFeatures,
                                                                               returnSeeds, subsetCategoryColumns):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fileName = f"{returnSeed['seed']}_{timestamp}.csv"
            folderPath = f"Data/Datasets/Input/Subsets/{seed['name']}"
            filePath = f"{folderPath}/{fileName}"

            os.makedirs(folderPath, exist_ok=True)
            subset.to_csv(filePath, index=False)

            metaFeatureDataset.append({
                "dataset_name": seed["name"],
                "seed": returnSeed["seed"],
                "subset_type": returnSeed["subsetType"],
                "file_name": filePath,
                **metaFeature
            })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(metaFeatureDataset).to_csv(f"Data/Datasets/Output/Raw/SubsetMetaFeatures_{timestamp}.csv", index=False)

def recreateDataset(subsetDataset, datasetNames, indexes, settingsFilePath, outputPath, numberOfFolds):
    dataset, outputPath = loadMetaFeaturesDataset(outputPath)
    settings = loadSettings(settingsFilePath)
    seeds = []
    for name, group in subsetDataset.groupby('dataset_name'):
        if name in datasetNames:
            seed = {
                "name": name,
                "rows": [],
            }
            for index, row in group.iterrows():
                seed["rows"].append({"seed": row['seed'], "subsetType": row['subset_type'], "file_path": row["file_name"], "index": index})
            seeds.append(seed)
    totalDuration = 0
    metaFeatures = subsetDataset.drop(columns=["dataset_name","seed","subset_type","file_name"])
    for seed in seeds:
        counter = 1
        for index in indexes:
            row = seed["rows"][index]
            trainingSet, testingSet, subsetCategoryColumns = loadSubset(row["file_path"], seed["name"], row["seed"])
            metaFeature = metaFeatures.iloc[row["index"]]
            instance, duration = createInstance(seed["name"], settings, numberOfFolds, trainingSet, testingSet, metaFeature, row, subsetCategoryColumns)
            totalDuration += duration
            dataset = pd.concat([dataset, instance], ignore_index=True)
            saveMetaFeaturesDataset(dataset, outputPath)
            predictedDuration = totalDuration/counter * (len(datasetNames))
            print(f"{counter} instance created from the {seed["name"]} dataset subset. It took {formatDuration(totalDuration)}/{formatDuration(predictedDuration)}")
            counter+=1

def createDataset(databaseName, outputPath, numberOfInstances, settingsFilePath, numberOfFolds, datasetSettings):
    dataset, outputPath = loadMetaFeaturesDataset(outputPath)
    settings = loadSettings(settingsFilePath)
    totalDuration = 0

    if numberOfInstances > 1:
        trainingSets, testingSets, metaFeatures, seeds, subsetCategoryColumns = createSubsets(databaseName, numberOfInstances, datasetSettings)
    else:
        trainingSets, testingSets, metaFeatures, seeds, subsetCategoryColumns = loadDataset(datasetSettings)

    counter = 0

    for trainingSet, testingSet, metaFeature, seed, categoryColumns in zip(trainingSets, testingSets, metaFeatures, seeds, subsetCategoryColumns):
        instance, duration = createInstance(databaseName, settings, numberOfFolds, trainingSet, testingSet, metaFeature, seed, categoryColumns)
        totalDuration += duration
        dataset = pd.concat([dataset, instance], ignore_index=True)
        saveMetaFeaturesDataset(dataset, outputPath)
        counter+=1
        predictedDuration = totalDuration/counter * numberOfInstances
        print(f"{counter} instance created. It took {formatDuration(totalDuration)}/{formatDuration(predictedDuration)}")
    return outputPath

def createInstance(datasetName, settings, numberOfFolds, trainingSet, testingSet, metaFeature, seed, categoryColumns):
    startTime = time.time()
    print("")
    print("Dataset name: "+datasetName)
    print("Seed: " + str(seed["seed"]))
    # Add dataset name, seed and meta feature
    instanceJSONObject= {"dataset_name": datasetName, "seed": seed["seed"], "subset_type": seed["subsetType"]}
    instanceJSONObject = {**instanceJSONObject, **metaFeature}
    bestTrainingLoss = float('inf')
    bestTrainingTechnique = ""
    bestTestingLoss = float('inf')
    bestTestingTechnique = ""
    random.seed(seed["seed"])
    seed = random.randint(0, 2**32 - 1)
    random.seed(seed)

    # Perform training for each configuration
    for config in configurations:
        print(config["param"])
        trainingLosses, testingLosses = trainNN(settings, config["param"], trainingSet, testingSet, seed, categoryColumns, numberOfFolds)

        instanceJSONObject[config['fileName']+"_training_loss"] = trainingLosses
        instanceJSONObject[config['fileName']+"_testing_loss"] = testingLosses

        if bestTrainingLoss > np.mean(trainingLosses):
            bestTrainingLoss = np.mean(trainingLosses)
            bestTrainingTechnique = config['fileName']

        if bestTestingLoss > np.mean(testingLosses):
            bestTestingLoss = np.mean(testingLosses)
            bestTestingTechnique = config['fileName']

    instanceJSONObject["best_training_technique"] = bestTrainingTechnique
    print("best training technique: "+bestTrainingTechnique)

    instanceJSONObject["best_testing_technique"] = bestTestingTechnique
    print("best testing technique: "+bestTestingTechnique)
    endTime = time.time()
    duration = endTime - startTime

    # Convert to DataFrame
    return pd.DataFrame([instanceJSONObject]), duration