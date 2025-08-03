import math

import torch
import torch.nn as nn


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # Calculate cross-entropy loss
        output = self.loss(predictions, targets)
        if math.isnan(output):
            raise Exception("Not a number")
        if math.isinf(output):
            raise Exception("Inf number")
        return output


class CustomCrossEntropyRegularisationTermLoss(nn.Module):
    def __init__(self, lambda_):
        super(CustomCrossEntropyRegularisationTermLoss, self).__init__()
        self.lambda_ = lambda_
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, model):
        l2RegularizationTerm = sum(torch.sum(param ** 2) for param in model.parameters())
        output = self.loss(predictions, targets) + 0.5 * self.lambda_ * l2RegularizationTerm
        if math.isnan(output):
            raise Exception("Not a number")
        if math.isinf(output):
            raise Exception("Inf number")
        return output
