import random

import pyhopper

from Utils.fileHandler import saveSettings, loadSettings
from Utils.menus import show_menu
from ModelTrainer.nnTrainer import trainNN
from Utils.datasetHandler import loadOptimiserDataset

maxNumberOfLayers = 10
minNumberOfLayers = 5
maxNumberOfEpoch = 1000

parameterGroups = ["All", "Basic", "Dropout", "Prune", "Weight decay", "Weight perturbation", "Back"]

basicParameters = {
    "batch_size": pyhopper.int(16, 1024, power_of=2),
    "learning_rate":  pyhopper.float(0.0001,0.5,"0.4f"),
    "momentum": pyhopper.float(0.0001,0.5,"0.4f"),
    "number_of_epochs": pyhopper.int(50, maxNumberOfEpoch, multiple_of=20),
    "number_of_hidden_layers": pyhopper.int(minNumberOfLayers, maxNumberOfLayers),
    "number_of_neurons_in_layers": pyhopper.int(5, 1000, multiple_of=5, shape=maxNumberOfLayers)
}

dropoutParameters = {
    "dropout_layers": pyhopper.float(0.01, 0.75, shape=maxNumberOfLayers)
}

pruneParameters = {
    "prune_amount": pyhopper.float(0.01, 0.75, "0.2f"),
    "prune_epoch_interval": pyhopper.int(4, 20)
}

weightDecayParameters = {
    "weight_decay": pyhopper.float(0.01, 0.5, "0.2f")
}

weightPerturbationParameters = {
    "weight_perturbation_amount": pyhopper.float(0.01, 1.00, "0.2f"),
    "weight_perturbation_interval": pyhopper.int(maxNumberOfEpoch/250, maxNumberOfEpoch/10)
}

datasetName = ""
basicSettings = {}
trainingSet = ""
validationSet = ""
categoryColumns = []
seed = random.randint(0, 4294967295)


def optimise_nn(datasetNameInput, datasetSettings):
    global datasetName, basicSettings, trainingSet, validationSet, categoryColumns
    parameterGroup = show_menu("Select parameter group by entering a number:", parameterGroups)
    if parameterGroup == parameterGroups[len(parameterGroups)-1]:
        return True
    datasetName = datasetNameInput
    sets, categoryColumns = loadOptimiserDataset(seed, datasetSettings)
    trainingSet = sets[0]
    validationSet = sets[1]
    while parameterGroup != parameterGroups[len(parameterGroups)-1]:
        if parameterGroup == parameterGroups[0]:
            bestParams = setupOptimiserAndRunIt(datasetName, "Basic", basicParameters, 300)
            path = saveSettings(bestParams, datasetName, "")
            basicSettings = loadSettings(path)

            bestParams = setupOptimiserAndRunIt(datasetName, "Dropout", dropoutParameters, 50)
            saveSettings(bestParams, datasetName, path)

            bestParams = setupOptimiserAndRunIt(datasetName, "Prune", pruneParameters, 100)
            saveSettings(bestParams, datasetName, path)

            bestParams = setupOptimiserAndRunIt(datasetName, "Weight_decay", weightDecayParameters, 50)
            saveSettings(bestParams, datasetName, path)

            bestParams = setupOptimiserAndRunIt(datasetName, "Weight_perturbation", weightPerturbationParameters, 100)
            saveSettings(bestParams, datasetName, path)
        else:
            if parameterGroup == parameterGroups[1]:
                bestParams = setupOptimiserAndRunIt(datasetName, parameterGroup, basicParameters, 300)
            else:
                basicSettings = loadSettings(input("Enter the path to the basic settings file of the NN:"))
                if parameterGroup == parameterGroups[2]:
                    bestParams = setupOptimiserAndRunIt(datasetName, parameterGroup, dropoutParameters, 50)
                elif parameterGroup == parameterGroups[3]:
                    bestParams = setupOptimiserAndRunIt(datasetName, parameterGroup, pruneParameters, 100)
                elif parameterGroup == parameterGroups[4]:
                    bestParams = setupOptimiserAndRunIt(datasetName, parameterGroup, weightDecayParameters, 50)
                else:
                    bestParams = setupOptimiserAndRunIt(datasetName, parameterGroup, weightPerturbationParameters, 100)
            saveSettings(bestParams, datasetName, "")
        parameterGroup = show_menu("Select parameter group by entering a number:", parameterGroups)
    return parameterGroup == parameterGroups[len(parameterGroups)-1]


def setupOptimiserAndRunIt(datasetName, parameterGroupName, parameterGroup, numberOfSteps):
    search = pyhopper.Search(parameterGroup)

    bestParams = search.run(
        trainNNWarp,
        direction="min",
        steps=numberOfSteps,
        checkpoint_path="Data/CheckPoints/" + datasetName.strip().replace(" ", "_") + "_"+ parameterGroupName + "_Checkpoint",
        # n_jobs="per-gpu"
    )
    testLoss = trainNNWarp(bestParams)
    print(f"Tuned params for {datasetName} dataset using {parameterGroupName} parameter group resulting in a of loss: {testLoss}")
    return bestParams


def trainNNWarp(params):
    global trainingSet, validationSet, categoryColumns
    if "batch_size" in params:
        settings = params
        trainingLosses, validationLosses = trainNN(settings, "", trainingSet, validationSet, categoryColumns, seed)
    else:
        settings = {**basicSettings, **params}
        if "dropout_layers" in params:
            trainingLosses, validationLosses = trainNN(settings, "dropout", trainingSet, validationSet, categoryColumns, seed)
        elif "prune_amount" in params:
            trainingLosses, validationLosses = trainNN(settings, "prune", trainingSet, validationSet, categoryColumns, seed)
        elif "weight_decay" in params:
            trainingLosses, validationLosses = trainNN(settings, "weightDecay", trainingSet, validationSet, categoryColumns, seed)
        else:
            trainingLosses, validationLosses = trainNN(settings, "weightPerturbation", trainingSet, validationSet,categoryColumns, seed)
    return validationLosses