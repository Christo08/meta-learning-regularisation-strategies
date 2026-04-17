from src.Utils.constants import *
from src.Utils.fileHandler import load_json_file, load_meta_features_csv
from src.Utils.metaFeatureDatasetHandler import prepare_meta_feature_dataset_for_states, prepare_meta_feature_sets


def show_menu(prompt, items):
    selection = -1
    while selection > len(items) - 1 or selection < 0:
        print(prompt)
        for database_name in items:
            print(str((items.index(database_name)+1))+". "+database_name)
        selection = int(input())-1
    return items[selection]

def show_dataset_menu(data_settings):
    dataset_names = ["All"]
    for datasetSettings in data_settings:
        dataset_names.append(datasetSettings["name"])
    dataset_names.append("Custom")
    dataset_names.append("Back")
    datasets_option = show_menu("Select dataset by entering a number: ", dataset_names)
    names = []
    if datasets_option == dataset_names[0]:
        names =  dataset_names[1:-2]
    elif datasets_option == dataset_names[len(dataset_names) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        for select_dataset_index in select_dataset_indexes:
            names.append(dataset_names[int(select_dataset_index) - 1])
    elif datasets_option != dataset_names[len(dataset_names) - 1]:
        names = [datasets_option]
    return names

def show_dataset_setting_menu():
    dataset_type = show_menu("Do you want to use the training datasets or testing datasets?", DATASET_TYPES)
    if dataset_type == DATASET_TYPES[2]:
        return False
    elif dataset_type == DATASET_TYPES[0]:
        datasets_setting_file_path = "Data/Datasets/Input/training_dataset_info.json"
    else:
        datasets_setting_file_path = "Data/Datasets/Input/testing_dataset_info.json"
    return load_json_file(datasets_setting_file_path)

def show_meta_leaner_type_menu():
    selected_meta_learn_types = show_menu("Select the meta-leaner type by entering its number: ", META_LEARN_TYPES)
    if selected_meta_learn_types == META_LEARN_TYPES[len(META_LEARN_TYPES) - 1]:
        return []
    elif selected_meta_learn_types == META_LEARN_TYPES[0]:
        selected_meta_learn_types = META_LEARN_TYPES[1:-2]
    elif selected_meta_learn_types ==  META_LEARN_TYPES[len(META_LEARN_TYPES) - 2]:
        print("Enter the meta-learner types' numbers separated by a comma:")
        selected_meta_learn_type_indexes = input().replace(' ', '').split(",")
        selected_meta_learn_types = []
        for selected_meta_learn_type_index in selected_meta_learn_type_indexes:
            selected_meta_learn_types.append(META_LEARN_TYPES[int(selected_meta_learn_type_index) - 1])
    else:
        selected_meta_learn_types = [selected_meta_learn_types]
    return selected_meta_learn_types

def show_dataset_loader_menu(allow_full_dataset = False, return_both_sets = False):
    if allow_full_dataset:
        was_processed= input("Has the dataset been processed before? (y/n): ").lower() == "y"
        set_types = ["Full dataset", "Training dataset", "Testing dataset"]
        set_type = show_menu("What type of dataset do you want to calculate stats for? ", set_types)
        if was_processed:
            if set_type == set_types[0]:
                return load_meta_features_csv()
            else:
                return load_meta_features_csv(set_type.split(" ")[0].strip().lower())
        else:
            if set_type == set_types[0]:
                return prepare_meta_feature_dataset_for_states()
            else:
                training_set, testing_set = prepare_meta_feature_sets()
                if set_type == set_types[1]:
                    return training_set
                else:
                    return testing_set
    elif return_both_sets:
            if input("Do you have training and testing sets? (y/n): ").lower() == "y":
                return load_meta_features_csv("training"), load_meta_features_csv("testing")
            else:
                return prepare_meta_feature_sets()
    else:
        if input("Do you have training sets? (y/n): ").lower() == "y":
            return load_meta_features_csv("training")
        else:
            training_set, _ = prepare_meta_feature_sets()
            return training_set