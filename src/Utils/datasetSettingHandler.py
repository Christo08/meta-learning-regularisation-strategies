from src.Utils.constants import DATASETS_INFO_PATH
from src.Utils.fileHandler import load_json_file
from src.Utils.menus import show_dataset_menu


class DatasetsSettingsHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatasetsSettingsHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.datasets_settings = load_json_file(DATASETS_INFO_PATH)
        if not self.datasets_settings:
            raise FileNotFoundError("No datasets settings found.")
        self._initialized = True

    def select_datasets_settings(self):
        names = show_dataset_menu(self.datasets_settings)
        selected_dataset_settings = []
        for name in names:
            selected_dataset_setting = next((item for item in self.datasets_settings if item["name"] == name), None)
            selected_dataset_settings.append(selected_dataset_setting)
        return selected_dataset_settings

    def select_dataset_name(self):
        return show_dataset_menu(self.datasets_settings)

    def get_dataset_settings(self):
        return self.datasets_settings