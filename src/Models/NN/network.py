import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.nn.utils.parametrizations import weight_norm


class Network(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 number_of_hidden_layers,
                 output_size,
                 dropout_layer=[],
                 batch_norm=False,
                 weight_norm_needed=False,
                 layer_norm=False,
                 activations=[]):
        super(Network, self).__init__()

        # Create a list to store the layers
        layers = []

        # Add the input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_sizes[0]))
        if len(activations) > 0:
            layers.append(activations[0])
        else:
            layers.append(nn.LeakyReLU())
        if len(dropout_layer) > 0 and dropout_layer[0] > 0:
            layers.append(nn.Dropout(p=dropout_layer[0]))

        # Add the hidden layers
        for index in range(number_of_hidden_layers - 1):
            layers.append(nn.Linear(hidden_sizes[index], hidden_sizes[index + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_sizes[index + 1]))
            if len(activations) > 0:
                layers.append(activations[index + 1])
            else:
                layers.append(nn.LeakyReLU())
            if len(dropout_layer) > 0 and dropout_layer[index+1] > 0:
                layers.append(nn.Dropout(p=dropout_layer[index+1]))

        if weight_norm_needed:
            for index in range(len(layers)):
                if isinstance(layers[index], nn.Linear):
                    layers[index] = weight_norm(layers[index], name='weight', dim=0)
        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[number_of_hidden_layers-1], output_size))
        layers.append(nn.Softmax(dim=1))

        # Combine all layers using Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def prune(self, pruning_method=prune.l1_unstructured, amount=0.2):
        for layer in self.network.children():
            if isinstance(layer, nn.Linear):
                pruning_method(layer, name='weight', amount=amount)

    def perturb_weights(self, perturbation_factor=0.01):
        for param in self.network.parameters():
            min_val = param.data.min()
            max_val = param.data.max()
            random_values = torch.rand_like(param) * (max_val - min_val) + min_val
            random_values = random_values * perturbation_factor
            if torch.cuda.is_available():
                random_values = random_values.cuda()
            param.data.add_(random_values)