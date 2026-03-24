from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader

from src.Models.NN.customDataset import CustomDataset
from src.Models.NN.lossFunctions import CustomCrossEntropyLoss
from src.Models.NN.network import Network
from src.Utils.constants import META_LEANER_TARGET_COLUMNS, MODULE_PATH
from src.Utils.datasetHandler import apply_smote
from src.Utils.fileHandler import load_settings, folder_maker
from src.Utils.metaLearnerStatsCalculator import MetaLearnerStats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_basic_nns(settings, technique, training_set, testing_set, seed, category_columns, fold=None):

    number_of_inputs = training_set[0].shape[1]
    number_of_outputs = training_set[1].shape[1]

    all_labels = training_set[1].columns.tolist()
    if fold is not None and fold >= 3:
        kf = KFold(n_splits=fold, shuffle=True, random_state=seed)

        training_mses = []
        training_accuracy = []

        testing_mses = []
        testing_accuracy = []

        counter =0

        for train_idx, _ in kf.split(training_set[0]):
            counter+=1
            training_set_x = training_set[0].iloc[train_idx]
            training_set_y = training_set[1].iloc[train_idx]
            training_set = (training_set_x, training_set_y)
            training_loss_value, training_accuracy_value, testing_loss_value, testing_accuracy_value = training_basic_loop(training_set,
                                                                                                                           testing_set,
                                                                                                                           settings,
                                                                                                                           number_of_inputs,
                                                                                                                           number_of_outputs,
                                                                                                                           technique,
                                                                                                                           seed,
                                                                                                                           all_labels,
                                                                                                                           category_columns)
            training_mses.append(training_loss_value)
            training_accuracy.append(training_accuracy_value)
            testing_mses.append(testing_loss_value)
            testing_accuracy.append(testing_accuracy_value)
        return training_mses, training_accuracy, testing_mses, testing_accuracy
    else:
        return training_basic_loop(training_set,
                                   testing_set,
                                   settings,
                                   number_of_inputs,
                                   number_of_outputs,
                                   technique,
                                   seed,
                                   all_labels,
                                   category_columns)

def training_basic_loop(training_set, testing_set, settings, number_of_inputs, number_of_outputs, technique, seed, all_labels, category_columns):
    global device
    x_training = training_set[0]
    y_training = training_set[1]
    if technique == "SMOTE":
        number_of_neighbors = number_of_outputs - 1
        if number_of_neighbors < 3 or x_training.shape[1] - len(category_columns) < 2:
            return float('inf'), float(0), float('inf'), float(0)
        try:
            x_training, y_training = apply_smote(x_training, y_training, seed, number_of_neighbors, category_columns)
        except Exception as e:
            if "Cannot apply smote." in str(e):
                return float('inf'), float(0), float('inf'), float(0)
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
    loss_function = CustomCrossEntropyLoss()
    if technique == "weightDecay":
        optimiser = optim.Adam(network.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    else:
        optimiser = optim.Adam(network.parameters(), lr=settings["learning_rate"])


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
        training_loss_value = loss_function(y_training_pred, y_training).item()
        testing_loss_value = loss_function(y_testing_pred, y_testing).item()
        # Classification accuracy: compare predicted class index vs true class index
        train_pred_cls = torch.argmax(y_training_pred, dim=1)
        train_true_cls = torch.argmax(y_training, dim=1)

        test_pred_cls = torch.argmax(y_testing_pred, dim=1)
        test_true_cls = torch.argmax(y_testing, dim=1)

        training_accuracy = (train_pred_cls == train_true_cls).float().mean().item() * 100.0
        testing_accuracy = (test_pred_cls == test_true_cls).float().mean().item() * 100.0

    return training_loss_value, training_accuracy, testing_loss_value, testing_accuracy


def training_meta_nns(settings_file_path, training_set, testing_set, seed, kFold =5):
    results = []
    settings = load_settings(settings_file_path)
    for target_column in META_LEANER_TARGET_COLUMNS:
        print(f"Training nn for { target_column.replace("_"," ")}...")
        training_x = np.array(training_set.drop([target_column], axis=1))
        training_y = training_set[target_column]
        testing_x = np.array(testing_set.drop([target_column], axis=1))
        testing_y = testing_set[target_column]
        result = train_meta_nn_loop(settings[target_column],
                                          (training_x, training_y),
                                          (testing_x, testing_y),
                                          seed,
                                          target_column,
                                          kFold)
        result = {
            "model type": "Neural Network",
            "technique": target_column.replace("_"," "),
            **result
        }
        results.append(result)
    return results

def train_meta_nn_loop(params, training_set, testing_set, seed, step, target_column ='na', kFold = 5):
    kf = KFold(n_splits=kFold, shuffle=True, random_state=seed)

    nn_stats = MetaLearnerStats()

    training_x = training_set[0]
    training_y = training_set[1]
    testing_x = testing_set[0]
    testing_y = testing_set[1]
    counter = 1
    print("")
    for train_idx, test_idx in kf.split(training_x):
        print(f"Fold {counter} step {step}")
        x_train = training_x[train_idx]
        y_train = training_y.iloc[train_idx].to_numpy()

        nn = train_nn((x_train, y_train), params)

        x_training = torch.tensor(x_train, dtype=torch.float32)
        y_training = torch.tensor(y_train, dtype=torch.float32)
        x_testing = torch.tensor(testing_x, dtype=torch.float32)
        y_testing_np = testing_y.to_numpy() if hasattr(testing_y, "to_numpy") else np.asarray(testing_y)
        y_testing = torch.tensor(y_testing_np, dtype=torch.float32)

        if torch.cuda.is_available():
            x_training = x_training.to(device)
            y_training = y_training.to(device)
            x_testing = x_testing.to(device)
            y_testing = y_testing.to(device)

        with torch.no_grad():
            y_train_pred = nn(x_training)
            y_test_pred = nn(x_testing)
        y_training_cpu = output_cleaner(y_training.detach().cpu().numpy())
        y_train_pred_cpu= output_cleaner(y_train_pred.detach().cpu().numpy())
        y_testing_cpu = output_cleaner(y_testing.detach().cpu().numpy())
        y_test_pred_cpu = output_cleaner(y_test_pred.detach().cpu().numpy())

        nn_stats.update_stats(y_training_cpu, y_train_pred_cpu, y_testing_cpu, y_test_pred_cpu)

        if target_column != 'na':
            folder_path = f"{MODULE_PATH}NN\\{datetime.now().strftime("%Y%m%d_%H")}"
            folder_maker(folder_path)
            checkpoint = {
                "model_class": "Network",
                "model_kwargs": {
                    "input_size": training_set[0].shape[1],
                    "hidden_sizes": params["number_of_neurons_in_layers"],
                    "number_of_hidden_layers": params["number_of_hidden_layers"],
                    "output_size": training_set[1].shape[1],
                },
                "state_dict": nn.state_dict(),
            }
            torch.save(checkpoint, f'{folder_path}\\nn_for_{target_column}_fold_{counter}.pt')
        counter = counter + 1
    return nn_stats.get_stats_json_object()

def train_nn(training_set, settings):
    global device

    number_of_inputs = training_set[0].shape[1]
    number_of_outputs = training_set[1].shape[1]

    # Convert data to tensors
    x_training = torch.tensor(training_set[0], dtype=torch.float32)
    y_training = torch.tensor(training_set[1], dtype=torch.float32)

    # Initialize the network based on the technique
    network = Network(input_size=number_of_inputs,
                      hidden_sizes=settings["number_of_neurons_in_layers"],
                      number_of_hidden_layers=settings["number_of_hidden_layers"],
                      output_size=number_of_outputs)

    # Move network and tensors to GPU if available
    if torch.cuda.is_available():
        network = network.to(device)
        x_training = x_training.to(device)
        y_training = y_training.to(device)

    # Create custom dataset and DataLoader
    train_dataset = CustomDataset(x_training, y_training)
    train_loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True)

    # Loss function and optimizer
    loss_function = CustomCrossEntropyLoss()
    optimiser = optim.Adam(network.parameters(), lr=settings["learning_rate"])


    # Training loop
    for epoch in range(settings["number_of_epochs"]):
        for batch in train_loader:
            network.train()
            x_batch = batch["data"]
            y_batch = batch["label"]

            training_outputs = network(x_batch)

            training_loss = loss_function(training_outputs, y_batch)

            optimiser.zero_grad()
            training_loss.backward()
            optimiser.step()
    return network

def output_cleaner(output):
    if output.ndim == 2:
        return np.argmax(output, axis=1)
    return output