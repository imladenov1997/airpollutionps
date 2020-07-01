from pathlib import Path
import json
import os


class ConfigReader:
    """
    Class for managing with config if such exists
    """
    CONFIG = None

    def __init__(self):
        pass

    @staticmethod
    def open_config(config='./airpyllution/config.json'):
        """
        Function for loading a config file, it must be JSON
        :param config: str - path to config
        :return: (True, None) | (False, str) - str is an error message
        """
        if not isinstance(config, str):
            return False, 'config input must be string'

        config_file = Path(config)
        if not config_file.is_file():
            return False, 'No such file'

        with open(config) as file:
            try:
                ConfigReader.CONFIG = json.load(file)
            except:
                return False, 'JSON error'

        return True, None
