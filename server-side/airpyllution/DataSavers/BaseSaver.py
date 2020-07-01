PREDICTOR_TYPE = 'ABSTRACT'


class AbstractSaver:
    def __init__(self, config=None):
        self.PREDICTOR_TYPE = PREDICTOR_TYPE
        self.config = self.__check_config(config)
        self.error = {
            'include': False
        }

    def save_predictions(self, X_test, predictions):
        pass

    def save_evaluations(self, X_test, predictions, target_values, metrics, error):
        pass

    def create_predictions_object(self, X_test, predictions, target=None):
        pass

    def add_error(self, metric, value):
        self.error['metric'] = metric
        self.error['value'] = value

        return self.error

    def __check_config(self, config):
        if not isinstance(config, dict):
            raise ValueError

        if 'pollutant' not in config or 'Pollutant' not in config['pollutant']:
            raise ValueError

        if 'predictionFile' not in config:
            raise ValueError

        return config