import random

import pyhopper

from ModelTrainer.decisionTreeTrainer import train_decision_tree
from Utils.fileHandler import saveModuleSettings
from Utils.metaFeatureDatasetHandler import createTestingSet, targetColumns

basicParameters = {
    "criterion": pyhopper.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
    "splitter": pyhopper.choice(["best", "random"]),
    "min_samples_split": pyhopper.int(10, 1000),
    "min_samples_leaf": pyhopper.int(10, 1000),
    "has_max_depth": pyhopper.choice([True, False]),
    "max_depth": pyhopper.int(5, 1000)
}

def optimise_decision_trees(dataset):
    global seed
    for targetColumn in targetColumns:
        seed = random.randint(0, 4294967295)
        optimiseDecisionTree(dataset, targetColumn)

def optimiseDecisionTree(fullDataset, targetColumn):
    global dataset, testing, seed, technique
    technique = targetColumn
    dataset, testing = createTestingSet(fullDataset, technique, seed)

    search = pyhopper.Search(basicParameters)
    bestParams = search.run(
        train,
        direction="min",
        steps=2500,
        checkpoint_path="Data/CheckPoints/DecisionTrees/"+technique+"_Decision_Tree_Checkpoint",
        n_jobs="per-gpu"
    )
    run, duration =  train_decision_tree(bestParams, dataset, testing, seed, technique)
    print(f"Tuned params for Decision Tree using basic parameter group resulting in a of loss: {run}")
    saveModuleSettings(bestParams, "DecisionTrees", technique)


def train(bestParams):
    global dataset, testing, seed, technique
    run, duration =  train_decision_tree(bestParams, dataset, testing, seed, technique, False)
    return run["validation_MSE"]