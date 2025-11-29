
def showMenu(prompt, items):
    selection = -1
    while selection > len(items) or selection < 0:
        print(prompt)
        for databaseName in items:
            print(str((items.index(databaseName)+1))+". "+databaseName)
        selection = int(input())-1
    return items[selection]

def showDatasetMenu(datasetNames):
    datasetsOption = showMenu("Select dataset by entering a number: ", datasetNames)
    if datasetsOption == datasetNames[0]:
        names =  datasetNames[1:-2]
    elif datasetsOption == datasetNames[len(datasetNames) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        selectDatasetIndexes = input().replace(' ', '').split(",")
        names = []
        for selectDatasetIndex in selectDatasetIndexes:
            names.append(datasetNames[int(selectDatasetIndex) - 1])
    elif datasetsOption == datasetNames[len(datasetNames) - 1]:
        return []
    else:
        names = [datasetsOption]
    return names