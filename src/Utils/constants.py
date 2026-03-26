#Paths
BASIC_NN_SETTINGS_PATH = "Models\\Settings\\BasicNN\\"
META_LEARNERS_SETTINGS_PATH = "Models\\Settings\\Meta-Learners\\"
DATASETS_INFO_PATH = "Data\\Input\\all_dataset_info.json"
MODULE_PATH ="Models\\"
SUBSET_PATH = "Data\\Input\\Subsets\\"
OUTPUT_PATH = "Data\\Output\\Raw\\"
RESULTS_PATH = "Data\\Results\\"
CHECK_POINTS_PATH = "Models\\CheckPoints\\"

#Menu items
PROCESS_OPTIONS = ["Optimise NN",  #0-1
                   "Create subsets and instances",  #1-2
                   "Recreate subsets",  #2-3
                   "Recreate instances",  #3-4
                   "Get statistics of meta learning dataset",  #4-5
                   "Split datasets into training and testing sets", #5-6
                   "Optimise meta learning",  #6-7
                   "Train meta learning",  #7-8
                   "Get statistics of meta learners results",  #8-9
                   "Exit"]
PARAMETER_GROUPS = ["All",
                    "Basic",
                    "Dropout",
                    "Prune",
                    "Weight decay",
                    "Weight perturbation",
                    "Back"]
DATASET_TYPES = ["Training",
                 "Testing",
                 "Back"]
META_LEARN_TYPES = ['All',
                    'Decision trees',
                    'K-nearest neighbors',
                    'Neural networks',
                    'Random forests',
                    'Support vector machines',
                    'Custom',
                    'Back']
STATS_OPTIONS = ["All",
                 "Bar charts of techniques' ranking count",
                 "Box charts of meta features vs techniques",
                 "Correlation heatmap of meta features vs techniques",
                 "Density plot of meta features",
                 "Features info",
                 "Ranking techniques info",
                 "Pair plot of meta features vs techniques",
                 "Custom",
                 "Back"]
OPTIMED_METRIC_OPTIONS = ["Accuracies",
                          "f1 scores",
                          "MSE"]

#SUBSET CREATION
MIN_CLASSES_REQUIRED = 2
MIN_INSTANCES_PER_SUBSET = 100
MIN_FEATURE_FRACTION = 0.5
OFFSET_RANGE_START = 1

#Other
REGULARISATION_TECHNIQUES = [
    {"name": "baseline", "param": "baseline", "fileName": "baseline"},
    {"name": "batchNormalisation", "param": "batchNormalisation", "fileName": "batch_normalisation"},
    {"name": "dropout", "param": "dropout", "fileName": "dropout"},
    {"name": "layerNormalisation", "param": "layerNormalisation", "fileName": "layer_normalisation"},
    {"name": "SMOTE", "param": "SMOTE", "fileName": "SMOTE"},
    {"name": "prune", "param": "prune", "fileName": "prune"},
    {"name": "weightDecay", "param": "weightDecay", "fileName": "weight_decay"},
    {"name": "weightNormalisation", "param": "weightNormalisation", "fileName": "weight_normalisation"},
    {"name": "weightPerturbation", "param": "weightPerturbation", "fileName": "weight_perturbation"}
]
TARGET_COLUMNS = ["baseline",
                  "batch_normalisation",
                  "dropout",
                  "layer_normalisation",
                  "prune",
                  "weight_normalisation",
                  "SMOTE",
                  "weight_decay",
                  "weight_perturbation"]

META_LEANER_TARGET_COLUMNS = ["baseline",
                              "batch_normalisation",
                              "dropout",
                              "layer_normalisation",
                              "prune",
                              "weight_normalisation",
                              "weight_decay",
                              "weight_perturbation"]