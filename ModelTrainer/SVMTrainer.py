import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def trainSVM(settings, dataset, testing, seed, showInfo = True):
    startTime = time.time()
    if showInfo:
        print("")
        print("Model Type: Support Vector Machines")
        print("Seed: " + str(seed))

    totalTrainingMSE = 0
    totalValidationMSE = 0
    totalTestingMSE = 0

    totalTrainingAccuracy = 0
    totalValidationAccuracy = 0
    totalTestingAccuracy = 0

    features = np.array(dataset[0])
    labels = np.array(dataset[1])

    kf = KFold(n_splits=settings["number_of_fold"], shuffle=True, random_state=seed)
    for fold, (trainIndex, testIndex) in enumerate(kf.split(dataset[0])):
        xTraining, xValidation = features[trainIndex], features[testIndex]
        yTraining, yValidation = labels[trainIndex], labels[testIndex]
        if settings["kernel"] == "linear":
            svm = SVC(C = settings["C"],
                      kernel = settings["kernel"],
                      random_state =seed)
        elif settings["kernel"] == "rbf" or settings["kernel"] == "sigmoid":
            if settings["gamma_type"] == "float":
                svm = SVC(C = settings["C"],
                          kernel = settings["kernel"],
                          gamma = settings["gamma"],
                          random_state =seed)
            else:
                svm = SVC(C = settings["C"],
                          kernel = settings["kernel"],
                          gamma = settings["gamma_type"],
                          random_state =seed)
        else:
            if settings["gamma_type"] == "float":
                svm = SVC(C = settings["C"],
                          kernel = settings["kernel"],
                          gamma = settings["gamma"],
                          degree =settings["degree"],
                          random_state =seed)
            else:
                svm = SVC(C = settings["C"],
                          kernel = settings["kernel"],
                          gamma = settings["gamma_type"],
                          degree =settings["degree"],
                          random_state =seed)

        svm = svm.fit(xTraining, yTraining)

        trainingPredict = svm.predict(xTraining)
        validationPredict = svm.predict(xValidation)
        testingPredict = svm.predict(testing[0])

        totalTrainingMSE += mean_squared_error(trainingPredict, yTraining)
        totalValidationMSE += mean_squared_error(validationPredict, yValidation)
        totalTestingMSE += mean_squared_error(testingPredict, testing[1])

        totalTrainingAccuracy += np.sum(yTraining == trainingPredict)/len(xTraining) * 100
        totalValidationAccuracy += np.sum(yValidation == validationPredict)/len(xTraining) * 100
        totalTestingAccuracy += np.sum(testing[1] == testingPredict)/len(testing[0]) * 100

    endTime = time.time()
    duration = endTime - startTime

    resultJSONObject = {"model_type": "Support Vector Machines",
                        "seed": seed,
                        "training_MSE": totalTrainingMSE/settings["number_of_fold"],
                        "validation_MSE": totalValidationMSE/settings["number_of_fold"],
                        "testing_MSE": totalTestingMSE/settings["number_of_fold"],
                        "training_accuracies": totalTrainingAccuracy/settings["number_of_fold"],
                        "validation_accuracies": totalValidationAccuracy/settings["number_of_fold"],
                        "testing_accuracies": totalTestingAccuracy/settings["number_of_fold"]}
    return resultJSONObject, duration