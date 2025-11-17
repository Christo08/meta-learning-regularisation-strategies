import torch

from InstanceCreator.instanceCreator import createDataset, recreateSubsets, recreateDataset
from ModelTrainer.metaLearningTrainer import metaLearningTrainer
from Optimisers.decisionTreeOptimiser import optimiseDecisionTrees
from Optimisers.nnOptimiser import optimiseNN
from Optimisers.randomForestOptimiser import optimiseRandomForest
from Optimisers.svmOptimiser import optimiseSVM
from Utils.createAvgNNSetting import createGenericNNSetting
from Utils.datasetStatsCalculator import calculateDatasetStats
from Utils.fileHandler import loadDatasetSetting, loadMetaFeaturesCSV
from Utils.menus import showMenu
from Utils.metaFeatureDatasetHandler import loadMetaFeatureDataset

datasetNames = ["All"]
processes = ["Optimise NN", #0
             "Create Avg NN Settings",#1
             "Recreate Subsets",#2
             "Recreate instances",#3
             "Create Subsets and instances",#4
             "Get Statistics of Meta Learning Dataset",#5
             "Optimise Meta Learning",#6
             "Train Meta Learning",#7
             "Exit"]
modulTypesNames = ["All", "Decision Tree", "Random Forest", "Support Vector Machines", "Back"]
datasetsSettings = {}

def main():
    global datasetsSettings
    print(f"PyTorch version: {torch.__version__}")  # Ensure it's a CUDA-compatible version
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version {torch.version.cuda}")
    else:
        print(f"Device: CPU")
    print(f"")
    datasetsSettings = loadDatasetSetting()
    for datasetSettings in datasetsSettings:
        datasetNames.append(datasetSettings["name"])
    datasetNames.append("Custom")
    datasetNames.append("Back")
    while True:
        process = showMenu("Select process by entering a number: ", processes)
        if process == processes[0]:
            while True:
                datasetName = showMenu("Select dataset by entering a number: ", datasetNames)
                if datasetName == datasetNames[0]:
                    for datasetSettings in datasetsSettings:
                        quited = optimiseNN(datasetSettings["name"])
                        if quited:
                            break
                elif datasetName == datasetNames[len(datasetNames) - 2]:
                    print("Select dataset by entering numbers separated by a comma:")
                    selectDatasetIndexes =  input().replace(' ', '').split(",")
                    selectDatasetNames = []
                    for selectDatasetIndex in selectDatasetIndexes:
                        selectDatasetNames.append(datasetNames[int(selectDatasetIndex)-1])
                    for datasetSettings in datasetsSettings:
                        if datasetSettings["name"] in selectDatasetNames:
                            quited = optimiseNN(datasetSettings["name"])
                            if quited:
                                break
        elif process == processes[1]:
            createGenericNNSetting()
        elif process == processes[2]:
            dataset = loadMetaFeatureDataset(True)
            numberOfInstances = 15  # int(input("How many Subsets do you what to create per dataset? "))
            recreateSubsets(dataset, numberOfInstances)
        elif process == processes[3]:
            datasetsOption = showMenu("Select dataset by entering a number: ", datasetNames)
            if datasetsOption == datasetNames[0]:
                names =  datasetNames[1:-2]
            elif datasetsOption == datasetNames[len(datasetNames) - 2]:
                selectDatasetIndexes = input().replace(' ', '').split(",")
                names = []
                for selectDatasetIndex in selectDatasetIndexes:
                    names.append(datasetNames[int(selectDatasetIndex) - 1])
            elif datasetsOption == datasetNames[len(datasetNames) - 1]:
                continue
            else:
                names = [datasetsOption]

            subsetDataset = loadMetaFeatureDataset(True)
            outputPath = "Data/Datasets/Output/Raw/regularisation_20251023_203810.csv"  # input("Enter the path of the Output dataset file or folder: ")
            settingsFilePath = "Data/Settings/NNSettings/Generic_nn_setting_20250811_073629.json"#input("Enter the path of the settings file: ")
            indexToCreate = [14]
            recreateDataset(subsetDataset, names, indexToCreate, settingsFilePath, outputPath)
        elif process == processes[4]:
            while True:
                datasetsOption = showMenu("Select dataset by entering a number: ", datasetNames)
                outputPath = "Data/Datasets/Output/Raw/regularisation_20251023_203810.csv" #input("Enter the path of the Output dataset file or folder: ")
                settingsFilePath = "Data/Settings/NNSettings/Generic_nn_setting_20250811_073629.json"#input("Enter the path of the settings file: ")
                numberOfInstances = 15#int(input("How many Subsets do you what to create per dataset? "))
                numberOfFolds = 5#int(input("How many folds do you what use per instance? "))
                if datasetsOption == datasetNames[0]:
                    for datasetSettings in datasetsSettings:
                        outputPath = createDataset(datasetSettings["name"], outputPath, numberOfInstances,
                                                   settingsFilePath, numberOfFolds)
                elif datasetsOption == datasetNames[len(datasetNames) - 2]:
                    print("Select dataset by entering numbers separated by a comma:")
                    selectDatasetIndexes = input().replace(' ', '').split(",")
                    for selectDatasetIndex in selectDatasetIndexes:
                        datasetName = datasetNames[int(selectDatasetIndex) - 1]
                        for datasetSettings in datasetsSettings:
                            if datasetSettings["name"] == datasetName:
                                outputPath = createDataset(datasetSettings["name"], outputPath, numberOfInstances,
                                                           settingsFilePath, numberOfFolds)
        elif process == processes[5]:
            runMetaFeatureDatasetProcess(calculateDatasetStats)
        elif process == processes[6]:
            runMetaFeatureDatasetProcess(optimiseMetaLearning)
        elif process == processes[7]:
            runMetaFeatureDatasetProcess(metaLearningTrainer)
        else:
            break

def runDatasetProcess(function):
    while True:
        datasetName = showMenu("Select dataset by entering a number: ", datasetNames)
        if datasetName == datasetNames[len(datasetNames) - 2]:
            print("Select dataset by entering numbers separated by a comma:")
            selectDatasetIndexes = input()
        if function.__name__ == "createDataset":
            outputPath = "Data/Datasets/Output/Raw/regularisation_20250923_071924.csv" #input("Enter the path of the Output dataset file or folder: ")
            settingsFilePath = "Data/Datasets/Output/Raw/regularisation_20250918_070808.csv"#input("Enter the path of the settings file: ")
            numberOfInstances = 15#int(input("How many Subsets do you what to create per dataset? "))
            numberOfFolds = 5#int(input("How many folds do you what use per instance? "))
        if datasetName == datasetNames[0]:
            for datasetSettings in datasetsSettings:
                if function.__name__ == "optimiseNN":
                    quited = function(datasetSettings["name"])
                else:
                    quited = function(datasetSettings["name"], outputPath, numberOfInstances, settingsFilePath, numberOfFolds)
                if quited:
                    break
        elif  datasetName == datasetNames[len(datasetNames) - 2]:
            selectDatasetIndexes = selectDatasetIndexes.replace(' ', '').split(",")
            selectDatasetNames = []
            for selectDatasetIndex in selectDatasetIndexes:
                selectDatasetNames.append(datasetNames[int(selectDatasetIndex)-1])
            for datasetSettings in datasetsSettings:
                if datasetSettings["name"] in selectDatasetNames:
                    if function.__name__ == "optimiseNN":
                        quited = function(datasetSettings["name"])
                    else:
                        quited = function(datasetSettings["name"], outputPath, numberOfInstances,settingsFilePath, numberOfFolds)
                    if quited:
                        break
        elif datasetName == datasetNames[len(datasetNames) - 1]:
            return
        else:
            if function.__name__ == "optimiseNN":
                function(datasetName)
            else:
                function(datasetName, outputPath, numberOfInstances,settingsFilePath, numberOfFolds)

def runMetaFeatureDatasetProcess(function):
    dataset = loadMetaFeatureDataset()
    function(dataset)

def optimiseMetaLearning(dataset):
    modulTypesName = showMenu("Select model type by entering a number: ", modulTypesNames)
    if modulTypesName == modulTypesNames[len(modulTypesNames)-1]:
        return
    elif modulTypesName == modulTypesNames[0]:
        optimiseDecisionTrees(dataset)
        optimiseRandomForest(dataset)
        optimiseSVM(dataset)
    elif modulTypesName == modulTypesNames[1]:
        optimiseDecisionTrees(dataset)
    elif modulTypesName == modulTypesNames[2]:
        optimiseRandomForest(dataset)
    else:
        optimiseSVM(dataset)


# Using the special variable
# __name__
if __name__=="__main__":
    main()