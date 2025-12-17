import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader

from Models.NN.customDataset import CustomDataset
from Models.NN.network import Network
from Utils.datasetHandler import applySMOTE
from Utils.lossFucntions import CustomCrossEntropyLoss, CustomCrossEntropyRegularisationTermLoss


def trainNN(settings, technique, trainingSet, testingSet, seed, categoryColumns, fold=None):

    numberOfInputs = trainingSet[0].shape[1]
    numberOfOutputs = trainingSet[1].shape[1]

    allLabels = trainingSet[1].columns.tolist()
    if fold is not None and fold >= 3:
        kf = KFold(n_splits=fold, shuffle=True, random_state=seed)
        trainingLossValues = []
        testingLossValues = []
        counter =0

        for train_idx, _ in kf.split(trainingSet[0]):
            counter+=1
            trainingSetX = trainingSet[0].iloc[train_idx]
            trainingSetY = trainingSet[1].iloc[train_idx]
            trainingLossValue, testingLossValue= trainingLoop(trainingSetX,
                                                              trainingSetY,
                                                              testingSet,
                                                              settings,
                                                              numberOfInputs,
                                                              numberOfOutputs,
                                                              technique,
                                                              seed,
                                                              allLabels,
                                                              categoryColumns)
            trainingLossValues.append(trainingLossValue)
            testingLossValues.append(testingLossValue)
        return trainingLossValues, testingLossValues
    else:
        trainingLossValues, testingLossValues= trainingLoop(trainingSet[0],
                            trainingSet[1],
                            testingSet,
                            settings,
                            numberOfInputs,
                            numberOfOutputs,
                            technique,
                            seed,
                            allLabels,
                            categoryColumns)
        return trainingLossValues, testingLossValues

def trainingLoop(xTraining, yTraining, testingSet, settings, numberOfInputs, numberOfOutputs, technique, seed, allLabels, categoryColumns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if technique == "SMOTE":
        numberOfNeighbors = numberOfOutputs - 1
        if numberOfNeighbors < 3 or xTraining.shape[1] - len(categoryColumns) < 2:
            return float('inf'), float('inf')
        try:
            xTraining, yTraining = applySMOTE(xTraining, yTraining, seed, numberOfNeighbors, categoryColumns)
        except Exception as e:
            if "Cannot apply smote." in str(e):
                return float('inf'), float('inf')
            else:
                raise e


        for label in allLabels:
            if label not in yTraining.columns:
                yTraining[label] = 0
        yTraining = yTraining[allLabels]

    # Convert data to tensors
    xTraining = torch.tensor(xTraining.values, dtype=torch.float32)
    yTraining = torch.tensor(yTraining.values, dtype=torch.float32)

    # Create custom dataset and DataLoader
    train_dataset = CustomDataset(xTraining, yTraining)
    train_loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)
    # Initialize the network based on the technique
    if technique == "batchNormalisation":
        network = Network(input_size=numberOfInputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=numberOfOutputs,
                          batch_norm=True)
    elif technique == "dropout":
        network = Network(input_size=numberOfInputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=numberOfOutputs,
                          dropout_layer=settings["dropout_layers"])
    elif technique == "layerNormalisation":
        network = Network(input_size=numberOfInputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=numberOfOutputs,
                          layer_norm=True)
    elif technique == "weightNormalisation":
        network = Network(input_size=numberOfInputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=numberOfOutputs,
                          weight_norm_needed=True)
    else:
        network = Network(input_size=numberOfInputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=numberOfOutputs)
    xTesting = torch.tensor(testingSet[0].values, dtype=torch.float32)
    yTesting = torch.tensor(testingSet[1].values, dtype=torch.float32)
    # Move network and tensors to GPU if available
    if torch.cuda.is_available():
        network = network.to(device)
        xTraining = xTraining.to(device)
        yTraining = yTraining.to(device)
        xTesting = xTesting.to(device)
        yTesting = yTesting.to(device)


    # Loss function and optimizer
    if technique == "weightDecay":
        lossFunction = CustomCrossEntropyRegularisationTermLoss(settings["weight_decay"])
    else:
        lossFunction = CustomCrossEntropyLoss()

    optimiser = optim.SGD(network.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"])

    # Training loop
    for epoch in range(settings["number_of_epochs"]):
        for batch in train_loader:
            network.train()
            xBatch = batch["data"]
            yBatch = batch["label"]


            if xBatch.shape[0] == 1 and technique == "batchNormalisation":
                continue

            if torch.cuda.is_available():
                xBatch = xBatch.to(device)
                yBatch = yBatch.to(device)

            trainingOutputs = network(xBatch)
            for index in range(xBatch.shape[0]):
                batch = xBatch[index]
                output = trainingOutputs[index]
                if torch.isnan(batch).any() or torch.isnan(output).any():
                    print(f"NaN detected in sample index {index} of the batch, skipping this sample.")

            if torch.isnan(xBatch).any():
                print("NaN detected in training batch, skipping this batch.")
            if torch.isnan(trainingOutputs).any():
                print("NaN detected in training outputs, skipping this batch.")

            if technique == "weightDecay":
                trainingLoss = lossFunction(trainingOutputs, yBatch, network)
            else:
                trainingLoss = lossFunction(trainingOutputs, yBatch)

            optimiser.zero_grad()
            trainingLoss.backward()
            optimiser.step()

        # Perform specific techniques during training
        if technique == "weightPerturbation" and epoch % settings["weight_perturbation_interval"] == 0 and epoch != 0:
            network.perturb_weights(perturbation_factor=settings["weight_perturbation_amount"])
        if technique == "prune" and epoch % settings["prune_epoch_interval"] == 0 and epoch != 0:
            network.prune(amount=settings["prune_amount"])

    # Final loss computation on training, validation, and testing sets
    with torch.no_grad():
        if technique == "weightDecay":
            trainingLossValue = lossFunction(network(xTraining), yTraining, network).item()
            testingLossValue = lossFunction(network(xTesting), yTesting, network).item()
        else:
            trainingLossValue = lossFunction(network(xTraining), yTraining).item()
            testingLossValue = lossFunction(network(xTesting), yTesting).item()

    return trainingLossValue, testingLossValue