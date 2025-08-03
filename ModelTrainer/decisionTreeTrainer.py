import time

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor


def trainDecisionTree(settings, dataset, testingSet, seed, technique, showInfo = True, fold=None):
    startTime = time.time()
    if showInfo:
        print("")
        print("Model Type: Decision Tree")
        print("Technique: "+technique)
        print("Seed: " + str(seed))

    totalTrainingMSE = 0
    totalValidationMSE = 0
    totalTestingMSE = 0

    if fold is not None and fold >= 3:
        features = np.array(dataset[0])
        labels = np.array(dataset[1])
        kf = KFold(n_splits=fold, shuffle=True, random_state=seed)
        for fold, (trainIndex, testIndex) in enumerate(kf.split(features)):
            xTraining, xValidation = features[trainIndex], features[testIndex]
            yTraining, yValidation = labels[trainIndex], labels[testIndex]

            trainingMSE, validationMSE, testingMSE = trainingLoop((xTraining, yTraining), (xValidation, yValidation), testingSet, settings, seed)

            totalTrainingMSE += trainingMSE
            totalValidationMSE += validationMSE
            totalTestingMSE += testingMSE

        totalTrainingMSE /= fold
        totalValidationMSE /= fold
        totalTestingMSE /= fold
    else:
        xTraining, xValidation, yTraining, yValidation = train_test_split(
            dataset[0], dataset[1], test_size=0.2, random_state=seed
        )
        totalTrainingMSE, totalValidationMSE, totalTestingMSE = trainingLoop((xTraining, yTraining), (xValidation, yValidation), testingSet, settings, seed)

    endTime = time.time()
    duration = endTime - startTime

    resultJSONObject = {
        "model_type": "Decision Tree",
        "technique": technique,
        "seed": seed,
        "training_MSE": totalTrainingMSE,
        "validation_MSE": totalValidationMSE,
        "testing_MSE": totalTestingMSE
    }
    return resultJSONObject, duration

def trainingLoop(trainingSet, validationSet, testingSet, settings, seed):
    if settings["has_max_depth"]:
        decisionTree = DecisionTreeRegressor(criterion=settings["criterion"],
                                             splitter=settings["splitter"],
                                             min_samples_split=settings["min_samples_split"],
                                             min_samples_leaf=settings["min_samples_leaf"],
                                             max_depth=settings["max_depth"],
                                             random_state=seed)
    else:
        decisionTree = DecisionTreeRegressor(criterion=settings["criterion"],
                                             splitter=settings["splitter"],
                                             min_samples_split=settings["min_samples_split"],
                                             min_samples_leaf=settings["min_samples_leaf"],
                                             random_state=seed)

    decisionTree = decisionTree.fit(trainingSet[0], trainingSet[1])

    trainingPredict = decisionTree.predict(trainingSet[0])
    validationPredict = decisionTree.predict(validationSet[0])
    testingPredict = decisionTree.predict(testingSet[0])

    trainingMSE = mean_squared_error(trainingSet[1], trainingPredict)
    validationMSE = mean_squared_error(validationSet[1], validationPredict)
    testingMSE = mean_squared_error(testingSet[1], testingPredict)
    return trainingMSE, validationMSE, testingMSE