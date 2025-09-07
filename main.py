import torch

from InstanceCreator.instanceCreator import createDataset
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
processes = ["Optimise NN",
             "Create Avg NN Settings",
             "Create Instance",
             "Get Statistics of Meta Learning Dataset",
             "Optimise Meta Learning",
             "Train Meta Learning",
             "Exit"]
modulTypesNames = ["All", "Decision Tree", "Random Forest", "Support Vector Machines", "Back"]
datasetsSettings = {}

def main():
    global datasetsSettings
    datasetsSettings = loadDatasetSetting()
    for datasetSettings in datasetsSettings:
        datasetNames.append(datasetSettings["name"])
    datasetNames.append("Custom")
    datasetNames.append("Back")
    print(f"PyTorch version: {torch.__version__}")  # Ensure it's a CUDA-compatible version
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version {torch.version.cuda}")
    else:
        print(f"Device: CPU")
    print(f"")
    while True:
        process = showMenu("Select process by entering a number: ", processes)
        if process == processes[0]:
            runDatasetProcess(optimiseNN)
        elif process == processes[1]:
            createGenericNNSetting()
        elif process == processes[2]:
            runDatasetProcess(createDataset)
        elif process == processes[3]:
            runMetaFeatureDatasetProcess(calculateDatasetStats)
        elif process == processes[4]:
            runMetaFeatureDatasetProcess(optimiseMetaLearning)
        elif process == processes[5]:
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
            outputPath = input("Enter the path of the Output dataset file or folder: ")
            settingsFilePath = input("Enter the path of the settings file: ")
            numberOfInstances = int(input("How many subsets do you what to create per dataset? "))
            numberOfFolds = int(input("How many folds do you what use per instance? "))
        if datasetName == datasetNames[0]:
            for datasetSettings in datasetsSettings:
                if function.__name__ == "optimiseNN":
                    quited = function(datasetSettings["name"])
                else:
                    quited = function(datasetSettings["name"], outputPath, numberOfInstances,settingsFilePath, numberOfFolds)
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