import random
import pyhopper

from ModelTrainer.randomForestTrainer import trainRandomForest
from Utils.datasetHandler import loadMetaFeaturesDataset
from Utils.fileHandler import saveModuleSettings

basicParameters = {
    "criterion": pyhopper.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
    "n_estimators": pyhopper.int(10, 100),
    "has_max_depth": pyhopper.choice([True, False]),
    "max_depth": pyhopper.int(10, 100),
    "min_samples_split": pyhopper.int(10, 100),
    "min_samples_leaf": pyhopper.int(10, 100),
    "bootstrap": pyhopper.choice([True, False])
}

seed = random.randint(0, 4294967295)

def optimiseRandomForest():
    global dataset, testing, seed
    dataset, testing = loadMetaFeaturesDataset(seed, False)
    search = pyhopper.Search(basicParameters)
    bestParams = search.run(
        train,
        direction="min",
        steps=2500,
        checkpoint_path="Data/CheckPoints/Random_Forest_Checkpoint",
        n_jobs="per-gpu"
    )
    run, duration =  trainRandomForest(bestParams, dataset, testing, seed)
    print(f"Tuned params for Random Forest using basic parameter group resulting in a of loss: {run}")
    saveModuleSettings(bestParams, "random_forest")


def train(bestParams):
    global dataset, testing, seed
    seed = random.randint(0, 4294967295)
    bestParams["number_of_fold"] = 2
    run, duration =  trainRandomForest(bestParams, dataset, testing, seed, False)
    return run["validation_MSE"]