from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

from src.Utils.statsCalculator import tp_tn_fp_fn


class MetaLearnerStats:

    def __init__(self):
        self.training_mses = []
        self.training_f1 = []
        self.training_accuracy = []

        self.testing_mses = []
        self.testing_f1 = []
        self.testing_accuracy = []
        self.testing_true_negative = 0
        self.testing_false_negative = 0
        self.testing_true_positive = 0
        self.testing_false_positive = 0

    def update_stats(self, y_training, y_train_pred, y_testing, y_test_pred):
        self.training_mses.append(mean_squared_error(y_training, y_train_pred))
        self.training_f1.append(f1_score(y_training, y_train_pred, average='weighted'))
        self.training_accuracy.append(accuracy_score(y_training, y_train_pred) * 100)

        self.testing_mses.append(mean_squared_error(y_testing, y_test_pred))
        self.testing_f1.append(f1_score(y_testing, y_test_pred, average='weighted'))
        self.testing_accuracy.append(accuracy_score(y_testing, y_test_pred)*100)
        self.testing_true_negative, self.testing_false_negative, self.testing_true_positive, self.testing_false_positive = tp_tn_fp_fn(y_testing, y_test_pred)

    def get_stats_json_object(self):
        return {
        "training loses": self.training_mses,
        "training accuracies": self.training_accuracy,
        "training f1": self.training_f1,
        "testing loses": self.testing_mses,
        "testing f1": self.testing_f1,
        "testing accuracies": self.testing_accuracy,
        "testing true positives": self.testing_true_positive,
        "testing true negatives": self.testing_true_negative,
        "testing false positives": self.testing_false_positive,
        "testing false negatives": self.testing_false_negative
    }