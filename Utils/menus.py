from Utils.fileHandler import load_dataset_setting_file


meta_learn_types = ['All', 'Decision trees', 'K-nearest neighbors', 'Neural networks', 'Random forests',
                    'Support vector machines', 'Custom', 'Back']
dataset_types = ["Training", "Testing", "Back"]

def show_menu(prompt, items):
    selection = -1
    while selection > len(items) or selection < 0:
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
    if datasets_option == dataset_names[0]:
        names =  dataset_names[1:-2]
    elif datasets_option == dataset_names[len(dataset_names) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        names = []
        for select_dataset_index in select_dataset_indexes:
            names.append(dataset_names[int(select_dataset_index) - 1])
    elif datasets_option == dataset_names[len(dataset_names) - 1]:
        return False
    else:
        names = [datasets_option]
    return names

def show_dataset_setting_menu():
    dataset_type = show_menu("Do you want to use the training datasets or testing datasets?", dataset_types)
    if dataset_type == dataset_types[2]:
        return False
    elif dataset_type == dataset_types[0]:
        datasets_setting_file_path = "Data/Datasets/Input/training_dataset_info.json"
    else:
        datasets_setting_file_path = "Data/Datasets/Input/testing_dataset_info.json"
    return load_dataset_setting_file(datasets_setting_file_path)

def show_meta_leaner_type_menu():
    selected_meta_learn_types = show_menu("Select the meta-leaner type by entering its number: ", meta_learn_types)
    if selected_meta_learn_types == meta_learn_types[len(meta_learn_types) - 1]:
        return []
    elif selected_meta_learn_types == meta_learn_types[0]:
        selected_meta_learn_types = meta_learn_types[1:-2]
    elif selected_meta_learn_types ==  meta_learn_types[len(meta_learn_types) - 2]:
        print("Enter the meta-learner types' numbers separated by a comma:")
        selected_meta_learn_type_indexes = input().replace(' ', '').split(",")
        selected_meta_learn_types = []
        for selected_meta_learn_type_index in selected_meta_learn_type_indexes:
            selected_meta_learn_types.append(meta_learn_types[int(selected_meta_learn_type_index) - 1])
    else:
        selected_meta_learn_types = [selected_meta_learn_types]
    return selected_meta_learn_types