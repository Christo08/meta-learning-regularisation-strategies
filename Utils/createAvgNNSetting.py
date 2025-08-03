import json
import os

from Utils.fileHandler import saveSettings


def createGenericNNSetting():
    settings = []
    genericNNSetting ={}
    print("Please enter a path to the folder which has the setting files:")
    folderPath = input().strip()

    if os.path.exists(folderPath) and os.path.isdir(folderPath):
        for file_name in os.listdir(folderPath):
            if file_name.endswith(".json") and not file_name.startswith("Generic"):
                file_path = os.path.join(folderPath, file_name)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        settings.append(data)
                except Exception as e:
                    print(f"Failed to load {file_name}: {e}")
        for setting in settings:
            for key in setting:
                if key not in genericNNSetting:
                    genericNNSetting[key] = setting[key]
                else:
                    if isinstance(setting[key], list):
                        if len(setting[key]) > len(genericNNSetting[key]):
                            for counter in range(len(setting[key])):
                                if counter < len(genericNNSetting[key]):
                                    genericNNSetting[key][counter]+= setting[key][counter]
                                else:
                                    genericNNSetting[key].append(setting[key][counter])
                        else:
                            for counter in range(len(setting[key])):
                                genericNNSetting[key][counter]+= setting[key][counter]
                    else:
                        genericNNSetting[key] += setting[key]
        for key in genericNNSetting:
            if isinstance(genericNNSetting[key], list):
                genericNNSetting[key] = [value / len(settings) for value in genericNNSetting[key]]
                if isinstance(settings[0][key][0], int):
                    genericNNSetting[key] = [round(value) for value in genericNNSetting[key]]
            elif genericNNSetting[key]:
                genericNNSetting[key] /= len(settings)
                if isinstance(settings[0][key], int):
                    genericNNSetting[key] = round(genericNNSetting[key])
        saveSettings(genericNNSetting, "Generic")
    else:
        print("Invalid folder path provided.")