
def showMenu(prompt, items):
    selection = -1
    while selection > len(items) or selection < 0:
        print(prompt)
        for databaseName in items:
            print(str((items.index(databaseName)+1))+". "+databaseName)
        selection = int(input())-1
    return items[selection]

def showDatasetMenu(prompt, items):
    selection = -1
    while selection > len(items) or selection < 0:
        print(prompt)
        for databaseName in items:
            print(str((items.index(databaseName)+1))+". "+databaseName)
        selection = int(input())-1
    return items[selection]