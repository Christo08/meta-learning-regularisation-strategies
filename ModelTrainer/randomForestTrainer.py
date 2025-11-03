import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold


def trainRandomForest(settings, dataset, testing, seed, showInfo = True):
    startTime = time.time()
    if showInfo:
        print("")
        print("Model Type: Random Forest")
        print("Seed: " + str(seed))

    totalTrainingMSE = 0
    totalValidationMSE = 0
    totalTestingMSE = 0

    totalTrainingMAE = 0
    totalValidationMAE = 0
    totalTestingMAE = 0

    totalTrainingR2 = 0
    totalValidationR2 = 0
    totalTestingR2 = 0

    features = np.array(dataset[0])
    labels = np.array(dataset[1])

    kf = KFold(n_splits=settings["number_of_fold"], shuffle=True, random_state=seed)
    for fold, (trainIndex, testIndex) in enumerate(kf.split(dataset[0])):
        xTraining, xValidation = features[trainIndex], features[testIndex]
        yTraining, yValidation = labels[trainIndex], labels[testIndex]

        if settings["has_max_depth"]:
            randomForest = RandomForestRegressor(n_estimators = settings["n_estimators"],
                                                  criterion = settings["criterion"],
                                                  max_depth = settings["max_depth"],
                                                  min_samples_split = settings["min_samples_split"],
                                                  min_samples_leaf = settings["min_samples_leaf"],
                                                  bootstrap = settings["bootstrap"],
                                                  random_state = seed)
        else:
            randomForest = RandomForestRegressor(n_estimators = settings["n_estimators"],
                                                  criterion = settings["criterion"],
                                                  min_samples_split = settings["min_samples_split"],
                                                  min_samples_leaf = settings["min_samples_leaf"],
                                                  bootstrap = settings["bootstrap"],
                                                  random_state = seed)

        randomForest = randomForest.fit(xTraining, yTraining)

        trainingPredict = randomForest.predict(xTraining)
        validationPredict = randomForest.predict(xValidation)
        testingPredict = randomForest.predict(testing[0])

        totalTrainingMSE += mean_squared_error(trainingPredict, yTraining)
        totalValidationMSE += mean_squared_error(validationPredict, yValidation)
        totalTestingMSE += mean_squared_error(testingPredict, testing[1])

        totalTrainingMAE += mean_absolute_error(yTraining, trainingPredict)
        totalValidationMAE += mean_absolute_error(yValidation, validationPredict)
        totalTestingMAE += mean_absolute_error(testing[1], testingPredict)

        totalTrainingR2 += r2_score(yTraining, trainingPredict)
        totalValidationR2 += r2_score(yValidation, validationPredict)
        totalTestingR2 += r2_score(testing[1], testingPredict)
    endTime = time.time()
    duration = endTime - startTime

    resultJSONObject = {"model_type": "Random Forest",
                        "seed": seed,
        "training_MSE": totalTrainingMSE / settings["number_of_fold"],
        "validation_MSE": totalValidationMSE / settings["number_of_fold"],
        "testing_MSE": totalTestingMSE / settings["number_of_fold"],
        "training_MAE": totalTrainingMAE / settings["number_of_fold"],
        "validation_MAE": totalValidationMAE / settings["number_of_fold"],
        "testing_MAE": totalTestingMAE / settings["number_of_fold"],
        "training_R2": totalTrainingR2 / settings["number_of_fold"],
        "validation_R2": totalValidationR2 / settings["number_of_fold"],
        "testing_R2": totalTestingR2 / settings["number_of_fold"]
    }
    return resultJSONObject, duration