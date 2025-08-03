import random
import time

import numpy as np
import pandas as pd

from ModelTrainer.nnTrainer import trainNN
from Utils.datasetHandler import createSubsets
from Utils.fileHandler import loadMetaFeaturesDataset, saveMetaFeaturesDataset, loadSettings
from Utils.timeFormatter import formatDuration

configurations = [
    {"name": "normal", "param": "normal", "fileName": "normal"},
    {"name": "batchNormalisation", "param": "batchNormalisation", "fileName": "batch_normalisation"},
    {"name": "dropout", "param": "dropout", "fileName": "dropout"},
    {"name": "layerNormalisation", "param": "layerNormalisation", "fileName": "layer_normalisation"},
    {"name": "SMOTE", "param": "SMOTE", "fileName": "SMOTE"},
    {"name": "prune", "param": "prune", "fileName": "prune"},
    {"name": "weightDecay", "param": "weightDecay", "fileName": "weight_decay"},
    {"name": "weightNormalisation", "param": "weightNormalisation", "fileName": "weight_normalisation"},
    {"name": "weightPerturbation", "param": "weightPerturbation", "fileName": "weight_perturbation"}
]

def createDataset(databaseName, outputPath, numberOfInstances, settingsFilePath, numberOfFolds):
    dataset, outputPath = loadMetaFeaturesDataset(outputPath)
    settings = loadSettings(settingsFilePath)
    totalDuration = 0

    trainingSets, testingSets, metaFeatures, seeds, subsetCategoryColumns = createSubsets(databaseName, numberOfInstances)
    counter = 0

    for trainingSet, testingSet, metaFeature, seed, categoryColumns in zip(trainingSets, testingSets, metaFeatures, seeds, subsetCategoryColumns):
        instance, duration = createInstance(databaseName, settings, numberOfFolds, trainingSet, testingSet, metaFeature, seed, categoryColumns)
        totalDuration += duration
        dataset = pd.concat([dataset, instance], ignore_index=True)
        saveMetaFeaturesDataset(dataset, outputPath)
        counter+=1
        predictedDuration = totalDuration/counter * numberOfInstances
        print(f"{counter} instance created. It took {formatDuration(totalDuration)}/{formatDuration(predictedDuration)}")

def createInstance(datasetName, settings, numberOfFolds, trainingSet, testingSet, metaFeature, seed, categoryColumns):
    startTime = time.time()
    print("")
    print("Dataset name: "+datasetName)
    print("Seed: " + str(seed))
    # Add dataset name, seed and meta feature
    instanceJSONObject= {"dataset_name": datasetName, "seed": seed}
    instanceJSONObject = {**instanceJSONObject, **metaFeature}
    bestTrainingLoss = float('inf')
    bestTrainingTechnique = ""
    bestTestingLoss = float('inf')
    bestTestingTechnique = ""
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