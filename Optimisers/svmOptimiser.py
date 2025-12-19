import random

import pyhopper

from ModelTrainer.SVMTrainer import trainSVM
from Utils.datasetHandler import loadMetaFeaturesDataset
from Utils.fileHandler import saveModuleSettings

basicParameters = {
    "C": pyhopper.choice([0.001,0.01,0.1,1,10,100]),
    "kernel": pyhopper.choice(['linear','rbf','poly','sigmoid']),
    "gamma_type": pyhopper.choice(['scale', 'auto', 'float', 'float']),
    "gamma": pyhopper.float(0.0001,100,log=True),
    "degree": pyhopper.int(1, 10)
}
seed = random.randint(0, 4294967295)

def optimise_svm():
    global dataset, testing, seed
    dataset, testing = loadMetaFeaturesDataset(seed, True)
    search = pyhopper.Search(basicParameters)
    bestParams = search.run(
        training,
        direction="min",
        steps=2500,
        checkpoint_path="Data/CheckPoints/SVM_Checkpoint",
        n_jobs="per-gpu"
    )
    run, duration =  trainSVM(bestParams, dataset, testing, seed)
    print(f"Tuned params for SVM using basic parameter group resulting in a of loss: {run}")
    saveModuleSettings(bestParams, "svm")


def training(bestParams):
    global dataset, testing, seed
    seed = random.randint(0, 4294967295)
    bestParams["number_of_fold"] = 2
    run, duration =  trainSVM(bestParams, dataset, testing, seed, False)
    return run["validation_MSE"]