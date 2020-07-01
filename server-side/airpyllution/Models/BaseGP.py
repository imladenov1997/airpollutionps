import json
import numpy as np

from airpyllution.Models.Exceptions import WrongNumberOfFeatures
from ..Metrics.Metrics import Metrics
from .BaseModel import AbstractBaseModel
from ..Utils.Errors import Errors
from abc import ABC, abstractmethod


class BaseGP(AbstractBaseModel):
    """
    Base Gaussian Processes Model, does not make assumptions whether it is Full or Sparse GP, builds on GPy
    """
    VARIANCE = None
    LENGTHSCALE = None
    TYPE = None
    RESOURCE = 'GPy'
    MODEL = None
    PATH = None

    def __init__(self):
        self.model = None
        self.kernel = None
        self.n_features = None
        self.stats = {
            'n_instances_trained': 0,
            'dataset_stats': {}
        }

    @abstractmethod
    def _init_kernel(self):
        pass

    def train(self, X_train, y_train, stats=None):
        """
        Function for training a model given a training dataset and pollution levels
        :param X_train: DataFrame - training set
        :param y_train: DataFrame - pollution values corresponding to instances of X_train
        :param stats: dict - stats that were generated when normalizing X
        :return:
        """
        input_dim = X_train.shape[1]

        # Initialise kernel if model was just created, if model was loaded kernel would have been already initialised
        if self.kernel is None:
            self._init_kernel(input_dim)

        if self.stats['n_instances_trained'] == 0:
            self.stats['n_instances_trained'] = X_train.shape[0]
            self.n_features = X_train.shape[1]
            self.stats['dataset_stats'] = stats
        elif self.__check_features(X_train):
            self.update_stats(stats, X_train.shape[0])
        else:
            raise WrongNumberOfFeatures('Training set does not have the same set of features model has been trained')

        self.model = self.MODEL(np.array(X_train), y_train, self.kernel)

    def predict(self, X_test, uncertainty=False):
        """
        Function for the model making predictions after being trained
        :param X_test: dataset to make predictions of pollution levels on
        :param uncertainty: bool - whether to return uncertainty along with predictions
        :return:
        """
        predictions = None

        if not self.__check_features(X_test):
            raise WrongNumberOfFeatures('Set does not have the same set of features model has been trained')

        if not isinstance(uncertainty, bool):
            uncertainty = False

        if uncertainty:
            predictions = self.predict_with_uncertainty(X_test)
        else:
            predictions = self.predict_without_uncertainty(X_test)

        return predictions

    def predict_with_uncertainty(self, X_test):
        """
        Function that makes predictions and returns uncertainty of predictions
        :param X_test: dataset with instances to predict pollution values
        :return: list with tuples, first element - predicted value, second element - uncertainty for the same prediction
        """
        predictions, y_std = self.model.predict(np.array(X_test))
        predictions = list(map(lambda x: x[0], predictions))
        y_std = list(map(lambda x: x[0], y_std))
        predictions_with_uncertainty = list(zip(predictions, y_std))
        return predictions_with_uncertainty

    def predict_without_uncertainty(self, X_test):
        """
        Function to make predictions without uncertainty
        :param X_test: DataFrame
        :return: list - predictions
        """
        predictions, _ = self.model.predict(np.array(X_test))
        predictions = list(map(lambda x: x[0], predictions))
        return predictions

    def eval(self, X_test, y_test, error_func=None):
        """
        Function for evaluating the accuracy of the model
        :param X_test: DataFrame - test dataset X
        :param y_test: DataFrame - pollution values corresponding to the instances in X_test
        :param error_func: function - should be a function that calculates certain error (RMSE, MAPE, etc.)
        :return:
        """
        predictions = self.predict(X_test)

        if not self.__check_features(X_test):
            raise WrongNumberOfFeatures('Set does not have the same set of features model has been trained')

        if error_func is None or not callable(error_func):
            error_func = Metrics.rmse  # default metrics

        result = error_func(y_test, predictions)

        return result, predictions, y_test

    def save_model(self, config):
        """"
        Save model's parameters in a separate file for further reuse of the model

        GPy suggests that it's better to save their models in JSON instead of serializing them with pickle as this is
        more consistent across python versions:
        https://github.com/SheffieldML/GPy/blob/devel/README.md

        The drawback in saving SparseGP is that it requires saving the dataset in the file as well as it is considered
        as part of the model, even with pickle it still requires saving data

        Model is loaded from SavedModels/GP/SparseGP and requires a single .json file

        :param config - dict from ConfigReader.CONFIG
        :return tuple - (is_successful, error_msg)
                        is_successful: boolean - whether the saving is successful or not
                        error_message: str|None - None when successful, otherwise shows what caused the error

        """
        if not isinstance(self.model, self.MODEL):
            return False, Errors.WRONG_INSTANCE.value

        if not isinstance(config, dict):
            return False, Errors.WRONG_CONFIG.value

        save_data = None

        if 'persistence' in config:
            save_data = config['persistence']
        else:
            return False, Errors.MODEL_NO_NAME.value

        if 'modelName' in save_data and isinstance(save_data['modelName'], str):
            path = './airpyllution/SavedModels/GP/' + self.PATH + save_data['modelName'] + '.json'
            model_params, extra_params = self.model_to_json(to_dict=False)  # extra_params are within model_params

            self.save_model_to_file(path, model_params)

            return True, None

        return False, Errors.NO_MODEL_DATA.value

    def load_model(self, config):
        """
        Function for loading a model from config
        :param config: dict - with details where model was located
        :return: (True, None) | (False, str) - str is the error message
        """
        if not isinstance(config, dict):
            return False, Errors.WRONG_CONFIG.value

        loading_data = None

        if 'loadedModel' in config:
            loading_data = config['loadedModel']
        else:
            return False, Errors.NO_SUCH_MODEL.value

        if 'modelName' in loading_data:
            path = './airpyllution/SavedModels/GP/' + self.PATH + loading_data['modelName'] + '.json'

            json_obj = self.load_model_from_file(path)

            if 'extra_params' in json_obj:
                if 'stats' in json_obj['extra_params'] and \
                   isinstance(json_obj['extra_params']['stats'], dict) and \
                   'n_instances_trained' in json_obj['extra_params']['stats'] and \
                   'dataset_stats' in json_obj['extra_params']['stats']:

                    self.stats = {
                        'n_instances_trained': json_obj['extra_params']['stats']['n_instances_trained'],
                        'dataset_stats': json_obj['extra_params']['stats']['dataset_stats']
                    }

            if 'data' in json_obj:
                if 'kernel' in json_obj['data'] and isinstance(json_obj['data']['kernel'], dict):
                    try:
                        self._init_kernel(None, custom=json_obj['data']['kernel'])
                    except:
                        return False, Errors.NO_VALID_KERNEL.value
                else:
                    return False, Errors.NO_KERNEL.value

                X = None
                y = None

                if 'datasets' in json_obj['data'] and isinstance(json_obj['data']['datasets'], dict):
                    X = np.array(json_obj['data']['datasets']['X']) if 'X' in json_obj['data']['datasets'] else None
                    y = np.array(json_obj['data']['datasets']['Y']) if 'Y' in json_obj['data']['datasets'] else None

                if X is None or y is None:
                    return False, Errors.NO_DATASETS_AVAILABLE.value

                if 'params' in json_obj['data']:
                    params = np.array(json_obj['data']['params'])

                    self.model = self.MODEL(X, y, self.kernel, initialize=False)
                    self.model.update_model(False)
                    self.model.initialize_parameter()
                    self.model[:] = params
                    self.model.update_model(True)

                    return True, None

                return False, Errors.NO_MODEL_PARAMS.value

            return False, Errors.NO_MODEL_DATA.value

        return False, Errors.NO_SUCH_MODEL.value

    @staticmethod
    def save_model_to_file(path, model):
        """
        Function to save model to file at a given path
        :param path: str - destination where model will be saved
        :param model: dict - model parameters in JSON
        :return:
        """
        with open(path, 'w+') as json_obj:
            json.dump(model, json_obj, sort_keys=False, indent=4)

    @staticmethod
    def load_model_from_file(path):
        """
        Function for loading a given model from file
        :param path: str - path to the model in the file
        :return:
        """
        with open(path, 'r') as file:
            json_obj = json.load(file)

        return json_obj

    def model_to_json(self, to_dict=True):
        """
        Generate model parameters in JSON
        :param to_dict: bool - whether it should be to dictionary or JSON
        :return: dict, dict | JSON, JSON
        """
        kernel = self.kernel.to_dict()
        model = {
            'type': self.TYPE,
            'resource': self.RESOURCE,
            'data': {
                'kernel': kernel,
                'params': self.model.param_array.tolist(),
                'datasets': {
                    'X': self.model.X.tolist(),
                    'Y': np.array(self.model.Y).tolist()
                }
            },
            'extra_params': {
                'n_features': self.n_features,
                'stats': self.stats
            }
        }

        if to_dict:
            return json.dumps(model), json.dumps(model['extra_params'])

        return model, self.stats

    def load_from_json(self, json_model_data, *args, **kwargs):
        """
        Load a model and its parameters from a JSON file, structure of the JSON must be the same as the same model was
        saved
        :param json_model_data: dict | JSON
        :param args: list
        :param kwargs: dict
        :return: (True, None) | (False, str) - str is error message
        """
        model_data = json_model_data if isinstance(json_model_data, dict) else json.loads(json_model_data)
        extra_params = args[0] if isinstance(args[0], dict) else json.loads(args[0])
        model_params = {}

        self.stats = {
            'n_instances_trained': extra_params['stats']['n_instances_trained'],
            'dataset_stats': extra_params['stats']['dataset_stats']
        }

        self.n_features = extra_params['n_features']

        if 'data' in model_data:
            model_params = model_data['data']
        else:
            return False, Errors.NO_MODEL_DATA.value

        if 'kernel' in model_params and isinstance(model_params['kernel'], dict):
            try:
                self._init_kernel(None, custom=model_params['kernel'])
            except:
                return False, Errors.NO_VALID_KERNEL.value
        else:
            return False, Errors.NO_KERNEL.value

        X = None
        y = None

        if 'datasets' in model_params and isinstance(model_params['datasets'], dict):
            X = np.array(model_params['datasets']['X']) if 'X' in model_params['datasets'] else None
            y = np.array(model_params['datasets']['Y']) if 'Y' in model_params['datasets'] else None

        if X is None or y is None:
            return False, Errors.NO_DATASETS_AVAILABLE.value

        if 'params' in model_params:
            params = np.array(model_params['params'])

            self.model = self.MODEL(X, y, self.kernel, initialize=False)
            self.model.update_model(False)
            self.model.initialize_parameter()
            self.model[:] = params
            self.model.update_model(True)

            return True, None

        return False, Errors.NO_MODEL_PARAMS.value

    @staticmethod
    @abstractmethod
    def new_from_json(json_obj):
        pass

    def update_stats(self, new_stats, n_new_instances):
        """
        Method for updating mean value of each feature in the dataset,
        it is important for predicting new datasets that have 0 mean by default
        :param new_stats: dict
        :param n_new_instances:
        :return:
        """
        n_cur_instances = self.stats['n_instances_trained']
        total = n_cur_instances + n_new_instances

        # Get weighted average of both
        weight_current_instances = n_cur_instances / total
        weight_new_instances = n_new_instances / total

        updated_dataset_stats = {}

        if 'dataset_stats' in self.stats and isinstance(self.stats['dataset_stats'], dict):
            for key, value in self.stats['dataset_stats'].items():
                if key in new_stats and 'mean' in new_stats[key] and 'std' in new_stats[key]:
                    updated_dataset_stats[key] = {}
                    updated_dataset_stats[key]['mean'] = new_stats[key]['mean'] * weight_new_instances
                    updated_dataset_stats[key]['std'] = new_stats[key]['std'] * weight_new_instances
                else:
                    print(key)
                    continue

                if 'mean' in value and 'std' in value:
                    updated_dataset_stats[key]['mean'] += value['mean'] * weight_current_instances
                    updated_dataset_stats[key]['std'] += value['std'] * weight_current_instances

        self.stats['n_instances_trained'] = total
        self.stats['dataset_stats'] = updated_dataset_stats

    def __check_features(self, new_dataset):
        """
        Compare features of the dataset model has been trained on and given dataset's features
        :param new_dataset: DataFrame
        :return: bool
        """
        if isinstance(self.stats, dict) and len(self.stats['dataset_stats']) != len(new_dataset.columns):
            return False

        for key in self.stats['dataset_stats'].keys():
            if key not in new_dataset:
                return False
        else:
            return True
