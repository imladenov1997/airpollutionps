from .BaseSaver import AbstractSaver

PREDICTOR_TYPE = 'CSV'


class CSVSaver(AbstractSaver):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.PREDICTOR_TYPE = PREDICTOR_TYPE

    def save_predictions(self, X_test, predictions):
        pass

    def save_evaluations(self, X_test, predictions, target_values, metrics, error):
        pass

    def create_predictions_object(self, X_test, predictions, target=None):
        pass

    @staticmethod
    def unnormalize(X, stats, inplace=False):
        pass

