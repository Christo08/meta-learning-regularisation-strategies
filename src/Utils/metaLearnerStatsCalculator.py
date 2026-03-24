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
        self.testing_true_positives = []
        self.testing_true_negatives = []
        self.testing_false_positives = []
        self.testing_false_negatives = []

    def update_stats(self, y_training, y_train_pred, y_testing, y_test_pred):
        self.training_mses.append(mean_squared_error(y_training, y_train_pred))
        self.training_f1.append(f1_score(y_training, y_train_pred, average='weighted'))
        self.training_accuracy.append(accuracy_score(y_training, y_train_pred) * 100)

        self.testing_mses.append(mean_squared_error(y_testing, y_test_pred))
        self.testing_f1.append(f1_score(y_testing, y_test_pred, average='weighted'))
        self.testing_accuracy.append(accuracy_score(y_testing, y_test_pred)*100)
        tp, tn, fp, fn = tp_tn_fp_fn(y_testing, y_test_pred)
        self.testing_true_positives.append(tp)
        self.testing_true_negatives.append(tn)
        self.testing_false_positives.append(fp)
        self.testing_false_negatives.append(fn)

    def get_stats_json_object(self):
        return {
        "training loses": self.training_mses,
        "training accuracies": self.training_accuracy,
        "training f1": self.training_f1,
        "testing loses": self.testing_mses,
        "testing f1": self.testing_f1,
        "testing accuracies": self.testing_accuracy,
        "testing true positives": self.testing_true_positives,
        "testing true negatives": self.testing_true_negatives,
        "testing false positives": self.testing_false_positives,
        "testing false negatives": self.testing_false_negatives
    }