from abc import ABC, abstractmethod


class AbstractBaseModel(ABC):
    TYPE = None
    RESOURCE = None

    @abstractmethod
    def train(self, X_train, y_train, stats=None):
        pass

    @abstractmethod
    def predict(self, X_test, uncertainty=False):
        pass

    @abstractmethod
    def predict_without_uncertainty(self, X_test):
        pass

    @abstractmethod
    def predict_with_uncertainty(self, X_test):
        pass

    @abstractmethod
    def eval(self, X_test, y_test, error_func=None):
        pass

    @abstractmethod
    def save_model(self, config):
        pass

    @abstractmethod
    def load_model(self, config):
        pass

    @abstractmethod
    def model_to_json(self, **kwargs):
        pass

    @abstractmethod
    def load_from_json(self, json_model_data, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def new_from_json(json_model_params, *args, **kwargs):
        pass

    @abstractmethod
    def update_stats(self, new_stats, n_new_instances):
        pass