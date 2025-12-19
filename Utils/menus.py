from Utils.fileHandler import load_dataset_setting_file


def show_menu(prompt, items):
    selection = -1
    while selection > len(items) or selection < 0:
        print(prompt)
        for databaseName in items:
            print(str((items.index(databaseName)+1))+". "+databaseName)
        selection = int(input())-1
    return items[selection]

def show_dataset_menu(dataSettings):
    datasetNames = ["All"]
    for datasetSettings in dataSettings:
        datasetNames.append(datasetSettings["name"])
    datasetNames.append("Custom")
    datasetNames.append("Back")
    datasetsOption = show_menu("Select dataset by entering a number: ", datasetNames)
    if datasetsOption == datasetNames[0]:
        names =  datasetNames[1:-2]
    elif datasetsOption == datasetNames[len(datasetNames) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        selectDatasetIndexes = input().replace(' ', '').split(",")
        names = []
        for selectDatasetIndex in selectDatasetIndexes:
            names.append(datasetNames[int(selectDatasetIndex) - 1])
    elif datasetsOption == datasetNames[len(datasetNames) - 1]:
        return False
    else:
        names = [datasetsOption]
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