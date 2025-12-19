import pandas as pd
import torch

from InstanceCreator.instanceCreator import createDataset, recreateSubsets, recreateDataset
from ModelTrainer.metaLearningTrainer import metaLearningTrainer
from Optimisers.decisionTreeOptimiser import optimiseDecisionTrees
from Optimisers.nnOptimiser import optimiseNN
from Optimisers.randomForestOptimiser import optimiseRandomForest
from Optimisers.svmOptimiser import optimiseSVM
from Utils.createAvgNNSetting import createGenericNNSetting
from Utils.datasetStatsCalculator import calculateDatasetStats
from Utils.metaFeatureDatasetHandler import loadMetaFeatureDataset
from Utils.menus import showDatasetMenu, showMenu, showDatasetSettingMenu

datasetNames = ["All"]
processes = ["Optimise NN", #0-1
             "Create Avg NN Settings",#1-2
             "Create Subsets and instances",#2-3
             "Recreate Subsets",#3-4
             "Recreate instances",#4-5
             "Get Statistics of Meta Learning Dataset",#5-6
             "Optimise Meta Learning",#6-7
             "Train Meta Learning",#7-8
             "Exit"]
modulTypesNames = ["All", "Random Forest", "Support Vector Machines", "NN", "Back"]

def main():
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
            while True:
                datasetsSettings = showDatasetSettingMenu()
                if not datasetsSettings:
                    break
                names = showDatasetMenu(datasetsSettings)
                if not names:
                    break
                for name in names:
                    datasetSettings = next((item for item in datasetsSettings if item["name"] == name), None)
                    quited = optimiseNN(name, datasetSettings)
                    if quited:
                         break
        elif process == processes[1]:
            createGenericNNSetting()
        elif process == processes[2]:
            while True:
                datasetsSettings = showDatasetSettingMenu()
                if not datasetsSettings:
                    break
                names = showDatasetMenu(datasetsSettings)
                if not names:
                    break
                outputPath = input("Enter the path of the Output dataset file or folder: ")
                settingsFilePath = input("Enter the path of the NN's settings file: ")
                numberOfInstances = int(input("How many Subsets do you what to create per dataset? "))
                numberOfFolds = int(input("How many folds do you what use per instance? "))
                for name in names:
                    datasetSettings = next((item for item in datasetsSettings if item["name"] == name), None)
                    outputPath = createDataset(name,
                                               outputPath,
                                               numberOfInstances,
                                               settingsFilePath,
                                               numberOfFolds,
                                               datasetSettings)
        elif process == processes[3]:
            datasetsSettings = showDatasetSettingMenu()
            if datasetsSettings:
                if input("Do you have a meta-feature file? (y/n): ").lower() == "y":
                    dataset = loadMetaFeatureDataset(True)
                    names =[]
                    numberOfInstances = int(input("How many Subsets do you what to create per dataset? "))
                    recreateSubsets(dataset, numberOfInstances, datasetsSettings, names)
                else:
                    dataset = pd.DataFrame(columns=["dataset_name","seed","number_of_features","proportion_of_numeric_features",
                                                    "number_of_instances","number_of_classes","ratio_of_instances_to_features",
                                                    "ratio_of_classes_to_features","ratio_of_instances_to_classes",
                                                    "ratio_of_min_to_max_instances_per_class","proportion_of_features_with_outliers",
                                                    "average_mutual_information","minimum_mutual_information",
                                                    "maximum_mutual_information","equivalent_number_of_features",
                                                    "noise_to_signal_ratio_of_features","baseline_training_loss",
                                                    "baseline_testing_loss","batch_normalisation_training_loss",
                                                    "batch_normalisation_testing_loss","dropout_training_loss","dropout_testing_loss",
                                                    "layer_normalisation_training_loss","layer_normalisation_testing_loss",
                                                    "SMOTE_training_loss","SMOTE_testing_loss","prune_training_loss","prune_testing_loss",
                                                    "weight_decay_training_loss","weight_decay_testing_loss","weight_normalisation_training_loss",
                                                    "weight_normalisation_testing_loss","weight_perturbation_training_loss",
                                                    "weight_perturbation_testing_loss","best_training_technique","best_testing_technique",
                                                    "subset_type"])
                    names = showDatasetMenu(datasetsSettings)
                    if names:
                        numberOfInstances = int(input("How many Subsets do you what to create per dataset? "))
                        recreateSubsets(dataset, numberOfInstances, datasetsSettings, names)
        elif process == processes[4]:
            datasetsSettings = showDatasetSettingMenu()
            if datasetsSettings:
                names = showDatasetMenu(datasetsSettings)
                if names:
                    subsetDataset = loadMetaFeatureDataset(True)
                    outputPath = input("Enter the path of the Output dataset file or folder: ")
                    settingsFilePath =input("Enter the path of the NN's settings file: ")
                    numberOfFolds = int(input("How many folds do you what use per instance? "))
                    indexToCreate = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
                    recreateDataset(subsetDataset, names, indexToCreate, settingsFilePath, outputPath, numberOfFolds)
        elif process == processes[5]:
            dataset = loadMetaFeatureDataset()
            calculateDatasetStats(dataset)
        elif process == processes[6]:
            modulTypesName = showMenu("Select model type by entering a number: ", modulTypesNames)
            if modulTypesName == modulTypesNames[len(modulTypesNames)-1]:
                return
            elif modulTypesName == modulTypesNames[0]:
                dataset = loadMetaFeatureDataset()
                optimiseDecisionTrees(dataset)
                optimiseRandomForest(dataset)
                optimiseSVM(dataset)
            elif modulTypesName == modulTypesNames[1]:
                dataset = loadMetaFeatureDataset()
                optimiseRandomForest(dataset)
            elif modulTypesName == modulTypesNames[2]:
                dataset = loadMetaFeatureDataset()
                optimiseSVM(dataset)
            else:
                optimiseSVM(dataset)
        elif process == processes[7]:
            dataset = loadMetaFeatureDataset()
            metaLearningTrainer(dataset)
        else:
            break

# Using the special variable
# __name__
if __name__=="__main__":
    main()