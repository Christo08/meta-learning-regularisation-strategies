import numpy as np
import torch
from numpy.f2py.auxfuncs import throw_error
from sklearn.metrics import mean_squared_error, accuracy_score, fbeta_score

from src.Utils.constants import OPTIMED_METRIC_OPTIONS
from src.Utils.statsCalculator import calculated_confusion_matrix


class MetaLearnerStats:
    # 50% recall and 50% Precision
    beta = 50/50

    def __init__(self, metric_type=OPTIMED_METRIC_OPTIONS[1]):
        self.metric_type = metric_type
        self.best_fold = -1
        self.modules = []

        self.training_mses = []
        self.training_f1 = []
        self.training_accuracy = []
        self.training_precision = []
        self.training_true_negative = []
        self.training_false_negative = []
        self.training_true_positive = []
        self.training_false_positive = []

        self.testing_mses = []
        self.testing_f1 = []
        self.testing_accuracy = []
        self.testing_precision = []
        self.testing_true_negative = []
        self.testing_false_negative = []
        self.testing_true_positive = []
        self.testing_false_positive = []

    def add_module(self, module):
        self.modules.append(module)

    def update_training_stats(self, y_training, y_train_pred):
        single_column_y_training = revert_encoding(y_training)
        single_column_y_training_pred = revert_encoding(y_train_pred)
        tp, tn, fp, fn = calculated_confusion_matrix(single_column_y_training, single_column_y_training_pred)

        self.training_mses.append(mean_squared_error(y_training, y_train_pred))
        self.training_f1.append(fbeta_score(single_column_y_training,
                                            single_column_y_training_pred,
                                            beta=self.beta,
                                            average='binary',
                                            pos_label=1))
        self.training_accuracy.append(accuracy_score(y_training, y_train_pred)*100)
        self.training_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        self.training_true_positive.append(tp)
        self.training_true_negative.append(tn)
        self.training_false_positive.append(fp)
        self.training_false_negative.append(fn)

    def update_testing_stats(self, y_testing, y_test_pred):
        single_column_y_testing = revert_encoding(y_testing)
        single_column_y_testing_pred = revert_encoding(y_test_pred)
        tp, tn, fp, fn = calculated_confusion_matrix(single_column_y_testing, single_column_y_testing_pred)

        self.testing_mses.append(mean_squared_error(y_testing, y_test_pred))
        self.testing_f1.append(fbeta_score(single_column_y_testing,
                                            single_column_y_testing_pred,
                                            beta=self.beta,
                                            average='binary',
                                            pos_label=1))
        self.testing_accuracy.append(accuracy_score(y_testing, y_test_pred)*100)
        self.testing_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        self.testing_true_positive.append(tp)
        self.testing_true_negative.append(tn)
        self.testing_false_positive.append(fp)
        self.testing_false_negative.append(fn)


    def get_training_stats_json_object(self):
        return {
            "training loses": self.training_mses if self.training_mses else 0.00,
            "training f1": self.training_f1 if self.training_f1 else 0.00,
            "training accuracies": self.training_accuracy if self.training_accuracy else 0.00,
            "training precision": self.training_precision if self.training_precision else 0.00,
            "training true positives": self.training_true_positive if self.training_true_positive else 0.00,
            "training true negatives": self.training_true_negative if self.training_true_negative else 0.00,
            "training false positives": self.training_false_positive if self.training_false_positive else 0.00,
            "training false negatives": self.training_false_negative if self.training_false_negative else 0.00
        }

    def get_testing_stats_json_object(self):
        return {
            "testing loses": self.testing_mses if self.testing_mses else 0.00,
            "testing f1": self.testing_f1 if self.testing_f1 else 0.00,
            "testing accuracies": self.testing_accuracy if self.testing_accuracy else 0.00,
            "testing precision": self.testing_precision if self.testing_precision else 0.00,
            "testing true positives": self.testing_true_positive if self.testing_true_positive else 0.00,
            "testing true negatives": self.testing_true_negative if self.testing_true_negative else 0.00,
            "testing false positives": self.testing_false_positive if self.testing_false_positive else 0.00,
            "testing false negatives": self.testing_false_negative if self.testing_false_negative else 0.00
        }

    def set_best_fold(self):
        if self.metric_type != OPTIMED_METRIC_OPTIONS[2]:
            best_metric = -1
        else:
            best_metric = float("inf")
        for counter, f1_score in enumerate(self.testing_f1):
            if self.metric_type == OPTIMED_METRIC_OPTIONS[0] and self.testing_accuracy[counter] > best_metric:
                best_metric = self.testing_accuracy[counter]
                self.best_fold = counter
            elif self.metric_type == OPTIMED_METRIC_OPTIONS[1] and f1_score > best_metric:
                best_metric = f1_score
                self.best_fold = counter
            elif self.metric_type == OPTIMED_METRIC_OPTIONS[2] and self.testing_mses[counter] < best_metric:
                best_metric = self.testing_mses[counter]
                self.best_fold = counter
            elif self.metric_type == OPTIMED_METRIC_OPTIONS[3] and  self.testing_precision[counter] > best_metric:
                best_metric = self.testing_precision[counter]
                self.best_fold = counter
            else:
               throw_error("Invalid metric type provided for best fold selection.")

    def get_best_fold(self):
        return self.best_fold

    def get_best_training_stats_json_object(self):
        if self.best_fold == -1:
            self.set_best_fold()
        return {
            "training loses": self.training_mses[self.best_fold] if self.training_mses[self.best_fold] else 0.00,
            "training f1": self.training_f1[self.best_fold] if self.training_f1[self.best_fold] else 0.00,
            "training accuracies": self.training_accuracy[self.best_fold] if self.training_accuracy[self.best_fold] else 0.00,
            "training precision": self.training_precision[self.best_fold] if self.training_precision[self.best_fold] else 0.00,
            "training true positives": self.training_true_positive[self.best_fold] if self.training_true_positive[self.best_fold] else 0.00,
            "training true negatives": self.training_true_negative[self.best_fold] if self.training_true_negative[self.best_fold] else 0.00,
            "training false positives": self.training_false_positive[self.best_fold] if self.training_false_positive[self.best_fold] else 0.00,
            "training false negatives": self.training_false_negative[self.best_fold] if self.training_false_negative[self.best_fold] else 0.00
        }

    def get_best_testing_stats_json_object(self):
        if self.best_fold == -1:
            self.set_best_fold()
        return {
            "testing loses": self.testing_mses[self.best_fold] if self.testing_mses[self.best_fold] else 0.00,
            "testing f1": self.testing_f1[self.best_fold] if self.testing_f1[self.best_fold] else 0.00,
            "testing accuracies": self.testing_accuracy[self.best_fold] if self.testing_accuracy[self.best_fold] else 0.00,
            "testing precision": self.testing_precision[self.best_fold] if self.testing_precision[self.best_fold] else 0.00,
            "testing true positives": self.testing_true_positive[self.best_fold] if self.testing_true_positive[self.best_fold] else 0.00,
            "testing true negatives": self.testing_true_negative[self.best_fold] if self.testing_true_negative[self.best_fold] else 0.00,
            "testing false positives": self.testing_false_positive[self.best_fold] if self.testing_false_positive[self.best_fold] else 0.00,
            "testing false negatives": self.testing_false_negative[self.best_fold] if self.testing_false_negative[self.best_fold] else 0.00
        }

    def get_best_model(self):
        if self.best_fold == -1:
            self.set_best_fold()
        return self.modules[self.best_fold]


def revert_encoding(encoded_tensor):
    if torch.is_tensor(encoded_tensor):
        return torch.argmax(encoded_tensor, dim=1)

    encoded = np.asarray(encoded_tensor)

    if encoded.ndim == 1:
        return encoded.astype(int)

    return np.argmax(encoded, axis=1).astype(int)
