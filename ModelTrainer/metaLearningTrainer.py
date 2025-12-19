import pandas as pd

from ModelTrainer.SVMTrainer import trainSVM
from ModelTrainer.decisionTreeTrainer import train_decision_tree
from ModelTrainer.randomForestTrainer import trainRandomForest
from Utils.datasetHandler import loadMetaFeaturesDataset
from Utils.fileHandler import loadRunsDataset, saveRunsDataset, loadModuleSettings
from Utils.menus import show_menu
from Utils.timeFormatter import format_duration

modelTypes = ["All", "Decision Tree", "Support Vector Machines", "Random Forest", "Exit"]

def meta_learning_trainer(dataset):
    runsPath = input("Enter the path of the runs file/folder :")
    dataset = loadMetaFeaturesDataset()
    runsDataset, fileName = loadRunsDataset(runsPath)
    if fileName !="":
        runsPath = runsPath +"\\"+ fileName
    modelType = 1
    while not modelType == modelTypes[len(modelTypes)-1]:
        modelType = show_menu("Select a model type:", modelTypes)
        if modelType == modelTypes[len(modelTypes)-1]:
            return
        if modelType == modelTypes[0] or modelType == modelTypes[1]:
            settingsPath =  input(f"Enter the path of the decision tree settings file :")
            decisionTreeSettings = loadModuleSettings(settingsPath)
        if modelType == modelTypes[0] or modelType == modelTypes[2]:
            settingsPath =  input(f"Enter the path of the Support Vector Machines settings file :")
            svmSettings = loadModuleSettings(settingsPath)
        if modelType == modelTypes[0] or modelType == modelTypes[3]:
            settingsPath =  input(f"Enter the path of the Random Forest settings file :")
            randomForestSettings = loadModuleSettings(settingsPath)
        numberOfInstances = int(input("How many time do you want to train each model? "))
        totalDuration = 0
        for counter in range(numberOfInstances):
            if modelType == modelTypes[0] or modelType == modelTypes[1]:
                run, duration = train_decision_tree(decisionTreeSettings, dataset)
                totalDuration += duration
                runsDataset = pd.concat([runsDataset, pd.DataFrame([run])])
                saveRunsDataset(runsPath, runsDataset)
            if modelType == modelTypes[0] or modelType == modelTypes[2]:
                run, duration = trainSVM(svmSettings, dataset)
                totalDuration += duration
                runsDataset = pd.concat([runsDataset, pd.DataFrame([run])])
                saveRunsDataset(runsPath, runsDataset)
            if modelType == modelTypes[0] or modelType == modelTypes[3]:
                run, duration = trainRandomForest(randomForestSettings, dataset)
                totalDuration += duration
                runsDataset = pd.concat([runsDataset, pd.DataFrame([run])])
                saveRunsDataset(runsPath, runsDataset)
            numberOfRuns = (counter + 1)
            if modelType == modelTypes[0]:
                numberOfRuns = numberOfRuns * (len(modelTypes) - 2)

            predictedDuration = totalDuration / numberOfRuns * numberOfInstances
            print(f"{numberOfRuns} runs created. It took {format_duration(totalDuration)}/{format_duration(predictedDuration)}")