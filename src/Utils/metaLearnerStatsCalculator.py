import numpy as np
import torch
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

from src.Utils.statsCalculator import calculated_confusion_matrix


class MetaLearnerStats:

    def __init__(self):
        self.training_mses = []
        self.training_f1 = []
        self.training_accuracy = []

        self.testing_mses = []
        self.testing_f1 = []
        self.testing_accuracy = []
        self.testing_true_negative = []
        self.testing_false_negative = []
        self.testing_true_positive = []
        self.testing_false_positive = []

    def update_stats(self, y_training, y_train_pred, y_testing, y_test_pred):
        self.training_mses.append(mean_squared_error(y_training, y_train_pred))
        single_column_y_training = revert_encoding(y_training)
        single_column_y_training_pred = revert_encoding(y_train_pred)
        self.training_f1.append(f1_score(single_column_y_training, single_column_y_training_pred, average='binary', pos_label=1))
        self.training_accuracy.append(accuracy_score(y_training, y_train_pred) * 100)

        self.testing_mses.append(mean_squared_error(y_testing, y_test_pred))
        single_column_y_testing =revert_encoding(y_testing)
        single_column_y_testing_pred = revert_encoding(y_test_pred)
        self.testing_f1.append(f1_score(single_column_y_testing, single_column_y_testing_pred, average='binary', pos_label=1))
        self.testing_accuracy.append(accuracy_score(y_testing, y_test_pred)*100)
        tp, tn, fp, fn = calculated_confusion_matrix(single_column_y_testing, single_column_y_testing_pred)
        self.testing_true_positive.append(tp)
        self.testing_true_negative.append(tn)
        self.testing_false_positive.append(fp)
        self.testing_false_negative.append(fn)

    def get_stats_json_object(self):
        return {
        "training loses": float(np.mean(self.training_mses)) if self.training_mses else 0.00,
        "training accuracies": float(np.mean(self.training_accuracy)) if self.training_mses else 0.00,
        "training f1": float(np.mean(self.training_f1)) if self.training_mses else 0.00,
        "testing loses": float(np.mean(self.testing_mses)) if self.training_mses else 0.00,
        "testing f1": float(np.mean(self.testing_f1)) if self.training_mses else 0.00,
        "testing accuracies": float(np.mean(self.testing_accuracy)) if self.training_mses else 0.00,
        "testing true positives": float(np.mean(self.testing_true_positive)) if self.training_mses else 0.00,
        "testing true negatives": float(np.mean(self.testing_true_negative)) if self.training_mses else 0.00,
        "testing false positives": float(np.mean(self.testing_false_positive)) if self.training_mses else 0.00,
        "testing false negatives": float(np.mean(self.testing_false_negative)) if self.training_mses else 0.00
    }

def revert_encoding(encoded_tensor):
    if torch.is_tensor(encoded_tensor):
        return torch.argmax(encoded_tensor, dim=1)

    encoded = np.asarray(encoded_tensor)

    if encoded.ndim == 1:
        return encoded.astype(int)

    return np.argmax(encoded, axis=1).astype(int)