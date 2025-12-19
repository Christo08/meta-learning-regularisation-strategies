from Utils.fileHandler import load_dataset_setting_file


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
    datasetTypes = ["Training", "Testing", "Back"]
    datasetType = show_menu("Do you want to use the training datasets or testing datasets?", datasetTypes)
    if datasetType == datasetTypes[2]:
        return False
    elif datasetType == datasetTypes[0]:
        datasetsSettingFilePath = "Data/Datasets/Input/training_dataset_info.json"
    else:
        datasetsSettingFilePath = "Data/Datasets/Input/testing_dataset_info.json"
    return load_dataset_setting_file(datasetsSettingFilePath)