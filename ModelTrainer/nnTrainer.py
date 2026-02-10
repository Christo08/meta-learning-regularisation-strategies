import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from Utils.datasetHandler import prepared_meta_feature_dataset
from Models.NN.customDataset import CustomDataset
from Models.NN.network import Network
from Utils.datasetHandler import apply_smote
from Utils.fileHandler import load_settings
from Utils.lossFunctions import CustomCrossEntropyLoss, CustomCrossEntropyRegularisationTermLoss

meta_learning_target_columns = ["baseline_testing_loss", "batch_normalisation_testing_loss", "dropout_testing_loss",
                                "layer_normalisation_testing_loss", "prune_testing_loss",
                                "weight_normalisation_testing_loss" ]


def train_nn(settings, technique, training_set, testing_set, seed, category_columns, fold=None):

    number_of_inputs = training_set[0].shape[1]
    number_of_outputs = training_set[1].shape[1]

    all_labels = training_set[1].columns.tolist()
    if fold is not None and fold >= 3:
        kf = KFold(n_splits=fold, shuffle=True, random_state=seed)
        training_loss_values = []
        training_accuracies_values = []
        testing_loss_values = []
        testing_accuracies_values = []
        counter =0

        for train_idx, _ in kf.split(training_set[0]):
            counter+=1
            training_set_x = training_set[0].iloc[train_idx]
            training_set_y = training_set[1].iloc[train_idx]
            training_loss_value, training_accuracy_value, testing_loss_value, testing_accuracy_value = training_loop(training_set_x,
                                                                                                                     training_set_y,
                                                                                                                     testing_set,
                                                                                                                     settings,
                                                                                                                     number_of_inputs,
                                                                                                                     number_of_outputs,
                                                                                                                     technique,
                                                                                                                     seed,
                                                                                                                     all_labels,
                                                                                                                     category_columns)
            training_loss_values.append(training_loss_value)
            training_accuracies_values.append(training_accuracy_value)
            testing_loss_values.append(testing_loss_value)
            testing_accuracies_values.append(testing_accuracy_value)
        return training_loss_values, training_accuracies_values, testing_loss_values, testing_accuracies_values
    else:
        return training_loop(training_set[0],
                             training_set[1],
                             testing_set,
                             settings,
                             number_of_inputs,
                             number_of_outputs,
                             technique,
                             seed,
                             all_labels,
                             category_columns)

def training_all_neural_networks(settings_file_path, raw_training_set, raw_testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    for target_column in meta_learning_target_columns:
        print(f"Training neural network for { target_column.replace("_testing_loss","").replace("_"," ")}...")
        training_set = prepared_meta_feature_dataset(raw_training_set, meta_learning_target_columns, target_column, False)
        testing_set = prepared_meta_feature_dataset(raw_testing_set, meta_learning_target_columns, target_column, False)

        training_y = training_set[1]
        testing_y = testing_set[1]

        if training_y.shape[1] > testing_y.shape[1]:
            difference = training_y.shape[1] - testing_y.shape[1]
            for i in range(difference):
                testing_y[f"{target_column}_class_{training_y.shape[1]-difference + i}"] = False
        if training_y.shape[1] < testing_y.shape[1]:
            difference = testing_y.shape[1] - training_y.shape[1]
            for i in range(difference):
                training_y[f"{target_column}_class_{testing_y.shape[1]-difference + i}"] = False

        training_set = (pd.DataFrame(training_set[0]), training_y)
        testing_set = (pd.DataFrame(testing_set[0]), testing_y)
        setting = settings[target_column]
        training_loss_values, training_accuracies_values, testing_loss_values, testing_accuracies_values = train_nn(setting, "", training_set, testing_set, seed, [], fold=kFold)
        result = {
            "model type": "NN",
            "technique": target_column.replace("_testing_loss","").replace("_"," "),
            "training losses": training_loss_values,
            "training accuracies": training_accuracies_values,
            "testing losses": testing_loss_values,
            "testing accuracies": testing_accuracies_values
        }
        results.append(result)
    return results

def training_loop(x_training, y_training, testing_set, settings, number_of_inputs, number_of_outputs, technique, seed, all_labels, category_columns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if technique == "SMOTE":
        number_of_neighbors = number_of_outputs - 1
        if number_of_neighbors < 3 or x_training.shape[1] - len(category_columns) < 2:
            return float('inf'), float('inf')
        try:
            x_training, y_training = apply_smote(x_training, y_training, seed, number_of_neighbors, category_columns)
        except Exception as e:
            if "Cannot apply smote." in str(e):
                return float('inf'), float('inf')
            else:
                raise e


        for label in all_labels:
            if label not in y_training.columns:
                y_training[label] = 0
        y_training = y_training[all_labels]

    # Convert data to tensors
    x_training = torch.tensor(x_training.values, dtype=torch.float32)
    y_training = torch.tensor(y_training.values, dtype=torch.float32)

    # Create custom dataset and DataLoader
    train_dataset = CustomDataset(x_training, y_training)
    train_loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)
    # Initialize the network based on the technique
    if technique == "batchNormalisation":
        network = Network(input_size=number_of_inputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=number_of_outputs,
                          batch_norm=True)
    elif technique == "dropout":
        network = Network(input_size=number_of_inputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=number_of_outputs,
                          dropout_layer=settings["dropout_layers"])
    elif technique == "layerNormalisation":
        network = Network(input_size=number_of_inputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=number_of_outputs,
                          layer_norm=True)
    elif technique == "weightNormalisation":
        network = Network(input_size=number_of_inputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=number_of_outputs,
                          weight_norm_needed=True)
    else:
        network = Network(input_size=number_of_inputs,
                          hidden_sizes=settings["number_of_neurons_in_layers"],
                          number_of_hidden_layers=settings["number_of_hidden_layers"],
                          output_size=number_of_outputs)
    x_testing = torch.tensor(testing_set[0].values, dtype=torch.float32)
    y_testing = torch.tensor(testing_set[1].values, dtype=torch.float32)
    # Move network and tensors to GPU if available
    if torch.cuda.is_available():
        network = network.to(device)
        x_training = x_training.to(device)
        y_training = y_training.to(device)
        x_testing = x_testing.to(device)
        y_testing = y_testing.to(device)


    # Loss function and optimizer
    if technique == "weightDecay":
        loss_function = CustomCrossEntropyRegularisationTermLoss(settings["weight_decay"])
    else:
        loss_function = CustomCrossEntropyLoss()

    optimiser = optim.SGD(network.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"])

    # Training loop
    for epoch in range(settings["number_of_epochs"]):
        for batch in train_loader:
            network.train()
            x_batch = batch["data"]
            y_batch = batch["label"]


            if x_batch.shape[0] == 1 and technique == "batchNormalisation":
                continue

            if torch.cuda.is_available():
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

            training_outputs = network(x_batch)
            for index in range(x_batch.shape[0]):
                batch = x_batch[index]
                output = training_outputs[index]
                if torch.isnan(batch).any() or torch.isnan(output).any():
                    print(f"NaN detected in sample index {index} of the batch, skipping this sample.")

            if torch.isnan(x_batch).any():
                print("NaN detected in training batch, skipping this batch.")
            if torch.isnan(training_outputs).any():
                print("NaN detected in training outputs, skipping this batch.")

            if technique == "weightDecay":
                training_loss = loss_function(training_outputs, y_batch, network)
            else:
                training_loss = loss_function(training_outputs, y_batch)

            optimiser.zero_grad()
            training_loss.backward()
            optimiser.step()

        # Perform specific techniques during training
        if technique == "weightPerturbation" and epoch % settings["weight_perturbation_interval"] == 0 and epoch != 0:
            network.perturb_weights(perturbation_factor=settings["weight_perturbation_amount"])
        if technique == "prune" and epoch % settings["prune_epoch_interval"] == 0 and epoch != 0:
            network.prune(amount=settings["prune_amount"])

    # Final loss computation on training, validation, and testing sets
    with torch.no_grad():
        y_training_pred = network(x_training)
        y_testing_pred = network(x_testing)
        if technique == "weightDecay":
            training_loss_value = loss_function(y_training_pred, y_training, network).item()
            testing_loss_value = loss_function(y_testing_pred, y_testing, network).item()
        else:
            training_loss_value = loss_function(y_training_pred, y_training).item()
            testing_loss_value = loss_function(y_testing_pred, y_testing).item()
        # Classification accuracy: compare predicted class index vs true class index
        train_pred_cls = torch.argmax(y_training_pred, dim=1)
        train_true_cls = torch.argmax(y_training, dim=1)
        training_accuracy = (train_pred_cls == train_true_cls).float().mean().item() * 100.0

        test_pred_cls = torch.argmax(y_testing_pred, dim=1)
        test_true_cls = torch.argmax(y_testing, dim=1)
        testing_accuracy = (test_pred_cls == test_true_cls).float().mean().item() * 100.0

    return training_loss_value, training_accuracy, testing_loss_value, testing_accuracy

def train_meta_nn(settings, training_set, testing_set, seed, target_column = 'na',fold=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert data to tensors
    x_training = torch.tensor(training_set[0], dtype=torch.float32)
    y_training = torch.tensor(training_set[1], dtype=torch.float32)

    # Create custom dataset and DataLoader
    train_dataset = CustomDataset(x_training, y_training)
    train_loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)
    # Initialize the network based on the technique
    network = Network(input_size=number_of_inputs,
                      hidden_sizes=settings["number_of_neurons_in_layers"],
                      number_of_hidden_layers=settings["number_of_hidden_layers"],
                      output_size=number_of_outputs)
    x_testing = torch.tensor(testing_set[0], dtype=torch.float32)
    y_testing = torch.tensor(testing_set[1], dtype=torch.float32)
    # Move network and tensors to GPU if available
    if torch.cuda.is_available():
        network = network.to(device)
        x_training = x_training.to(device)
        y_training = y_training.to(device)
        x_testing = x_testing.to(device)
        y_testing = y_testing.to(device)

    # Loss function and optimizer
    loss_function = CustomCrossEntropyLoss()

    optimiser = optim.SGD(network.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"])

    # Training loop
    for epoch in range(settings["number_of_epochs"]):
        for batch in train_loader:
            network.train()
            x_batch = batch["data"]
            y_batch = batch["label"]

            if torch.cuda.is_available():
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

            training_outputs = network(x_batch)
            for index in range(x_batch.shape[0]):
                batch = x_batch[index]
                output = training_outputs[index]
                if torch.isnan(batch).any() or torch.isnan(output).any():
                    print(f"NaN detected in sample index {index} of the batch, skipping this sample.")

            if torch.isnan(x_batch).any():
                print("NaN detected in training batch, skipping this batch.")
            if torch.isnan(training_outputs).any():
                print("NaN detected in training outputs, skipping this batch.")

            training_loss = loss_function(training_outputs, y_batch)

            optimiser.zero_grad()
            training_loss.backward()
            optimiser.step()

    # Final loss computation on training, validation, and testing sets
    with torch.no_grad():
        y_training_pred = network(x_training)
        y_testing_pred = network(x_testing)
        training_loss_value = loss_function(y_training_pred, y_training).item()
        testing_loss_value = loss_function(y_testing_pred, y_testing).item()
        # Classification accuracy: compare predicted class index vs true class index
        train_pred_cls = torch.argmax(y_training_pred, dim=1)
        train_true_cls = torch.argmax(y_training, dim=1)
        training_accuracy = (train_pred_cls == train_true_cls).float().mean().item() * 100.0

        test_pred_cls = torch.argmax(y_testing_pred, dim=1)
        test_true_cls = torch.argmax(y_testing, dim=1)
        testing_accuracy = (test_pred_cls == test_true_cls).float().mean().item() * 100.0

    return {
        "training loss": training_loss_value,
        "training accuracies": training_accuracy,
        "testing loss": testing_loss_value,
        "testing accuracies": testing_accuracy
    }