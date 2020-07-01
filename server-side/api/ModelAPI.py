import json

import math

import numpy
from pandas import Series

from airpyllution import ConvolutionalNeuralNetwork
from airpyllution import DBManager
from airpyllution import MainTransformer
from airpyllution import SparseGaussianProcesses
from airpyllution.Models.FullGP import GaussianProcesses
from airpyllution.Models import MODEL_TYPES
from utils.Errors import Errors
from .DatasetsAPI import DatasetsApi


class ModelApi:
    DATE_TIME_FORMAT = '%d-%m-%Y %H:%M'  # accepted format for datetime
    REQUIRED_FIELDS = {'Datetime', 'Longitude', 'Latitude', 'Pollutant', 'Uncertainty'}

    @staticmethod
    def create_model(name, body):
        """
        Function for creating a non-existing model and training it with a given dataset
        This function should happen in the background to prevent overhead to Flask
        :param name: unique name of the model
        :param body: dict with following data:
        * type - type of model (CNN, FullGP, etc.)
        * range - dict with start and end fields, each storing datetime in DATE_TIME_FORMAT
        * locations - list of lists, nested list should have two entries 0 - longitude, 1 - latitude
        * pollutant - name of the polllutant PM10, PM2.5
        * data - dict object with additional data that would be stored as JSONB data, it could have keys such as
        weather
        :return: bool: whether model was created
        """

        if body is None:
            return False, Errors.MISSING_BODY.value

        print('Getting dataset...')
        dataset = DatasetsApi.get_dataset(body, use_dataframe=True)
        print(dataset)

        if dataset is None:
            return False, Errors.NO_DATA.value

        model = None
        complete_dataset = dataset[dataset['Pollutant'].notnull()]

        X_train, y_train, _, _, stats = MainTransformer.get_training_and_test_set(complete_dataset,
                                                                                  'Pollutant',
                                                                                  'Uncertainty',
                                                                                  size=1,
                                                                                  normalize=True)

        if 'type' not in body:
            return False, Errors.NO_MODEL_TYPE_GIVEN.value

        if body['type'] == 'CNN':
            model = ConvolutionalNeuralNetwork()
            model.train(X_train, y_train, stats=stats)
            resource = 'keras'
            model_params, extra_params = model.model_to_json()
            result = DBManager.upsert_model(name, body['type'], resource, model_params=model_params,
                                            extra_params=extra_params)
            return True, None
        elif body['type'] == 'FullGP':
            model = GaussianProcesses()
            model.train(X_train, y_train, stats=stats)
            resource = 'GPy'
            model_params, extra_params = model.model_to_json()
            result = DBManager.upsert_model(name, body['type'], resource, model_params=model_params,
                                            extra_params=extra_params)
            return True, None
        elif body['type'] == 'SparseGP':
            model = SparseGaussianProcesses()
            model.train(X_train, y_train, stats=stats)
            resource = 'GPy'
            model_params, extra_params = model.model_to_json()
            result = DBManager.upsert_model(name, body['type'], resource, model_params=model_params,
                                            extra_params=extra_params)
            return True, None

        return False, Errors.NO_SUCH_MODEL_TYPE.value

    @staticmethod
    def make_predictions(body, overwrite=False):
        """
        Function for making predictions over a time range and locations by a given model
        :param body:
        :param overwrite:
        :return: bool, list - boolean whether it is successful and list with predictions and uncertanties
        """
        dataset = DatasetsApi.get_dataset(body, use_dataframe=True)

        if dataset is None:
            return False, []

        # get dataset with empty pollutant values
        incomplete_dataset = dataset if overwrite else dataset[dataset['Pollutant'].isnull()]

        # split the dataset, do not normalize until means and stds are taken from the model
        X_predict, y_predict, _, _, stats = MainTransformer.get_training_and_test_set(incomplete_dataset,
                                                                                      'Pollutant',
                                                                                      'Uncertainty',
                                                                                      size=1,
                                                                                      normalize=False)

        model, model_record, err = ModelApi.get_model_by_name(body['name'])
        predictions = []
        print(err)
        if err is None:
            training_dataset_stats = {}
            print('Verifying features...')
            if X_predict is None or X_predict.shape[1] != model.n_features:
                print('Wrong number of features')
                print(X_predict.shape[1] - 1)
                print(model.n_features)
                return False, []

            print('Checking model stats...')
            if 'dataset_stats' in model.stats:
                training_dataset_stats = model.stats['dataset_stats']
                feature_names = set(training_dataset_stats.keys())
                dataset_features = set(X_predict)
                dataset_features.discard('DateTime')

                print('Checking feature names...')
                if feature_names != dataset_features:
                    return False, []

                print('Normalizing...')
                MainTransformer.normalize(X_predict, stats=training_dataset_stats, inplace=True)
            else:
                return False, []

            print('Preidicting...')
            predictions = model.predict(X_predict, uncertainty=True)
            MainTransformer.unnormalize(X_predict, training_dataset_stats, inplace=True)
            MainTransformer.remove_periodic_f(X_predict)
            X_predict.loc[:, 'Pollutant'] = Series([x[0] for x in predictions], index=X_predict.index)
            X_predict.loc[:, 'Uncertainty'] = Series([x[1] for x in predictions], index=X_predict.index)
            # add predictions to the DB

            print('Done. Adding to database...')
            optional_data_keyset = set(body['data'].keys())
            dataframe_optional_data = set(X_predict.keys()).difference(ModelApi.REQUIRED_FIELDS)
            keys_with_data_to_be_added = optional_data_keyset.intersection(dataframe_optional_data)
            results = []
            for index, row in X_predict.iterrows():
                if row['Pollutant'] is not None and math.isnan(row['Pollutant']):
                    continue
                input_instance = {
                    'date_time': index,
                    'longitude': row['Longitude'],
                    'latitude': row['Latitude'],
                    'pollutant': body['pollutant'],
                    'pollution_value': row['Pollutant'],
                    'uncertainty': row['Uncertainty'],
                    'data': {}
                }

                print(body['pollutant'])
                print(row['Pollutant'])

                for key in keys_with_data_to_be_added:
                    input_instance['data'][key] = row[key]

                result = DatasetsApi.insert_single_prediction(input_instance)
                results.append(result)

            predictions = ModelApi.__predictions_to_primitive_float(predictions)
            print('failed following: ')
            print(list(filter(lambda x: not x[0], results)))

            return True, predictions

        return False, predictions  # in case that model does not exist

    @staticmethod
    def make_single_prediction(body):
        """
        Function for making predictions over a time range and locations by a given model
        :param body:
        :return: bool, list - boolean whether it is successful and list with predictions and uncertanties
        """

        if not isinstance(body, dict):
            return False, []

        if 'name' not in body:
            return False, []

        if 'pollutant' not in body:
            return False, []

        model, model_record, err = ModelApi.get_model_by_name(body['name'])
        predictions = []

        if err is None:
            prev = None
            if isinstance(model, ConvolutionalNeuralNetwork):
                prev = model.seq_length

            training_dataset_stats = {}
            if 'dataset_stats' in model.stats:
                training_dataset_stats = model.stats['dataset_stats']
                X_predict = DatasetsApi.get_single_instance_dataset(body, stats=training_dataset_stats, prev=prev)

                if X_predict is None:
                    return False, []

                feature_names = set(training_dataset_stats.keys())
                dataset_features = set(X_predict)
                dataset_features.discard('DateTime')

                if feature_names != dataset_features:
                    print(feature_names)
                    print(dataset_features)
                    return False, []
            else:
                return False, []

            predictions = model.predict(X_predict, uncertainty=True)
            MainTransformer.unnormalize(X_predict, training_dataset_stats, inplace=True)
            MainTransformer.remove_periodic_f(X_predict)
            X_predict.loc[:, 'Pollutant'] = Series([x[0] for x in predictions], index=X_predict.index)
            X_predict.loc[:, 'Uncertainty'] = Series([x[1] for x in predictions], index=X_predict.index)
            # add predictions to the DB

            keys_with_data_to_be_added = {}
            if 'data' in body:
                optional_data_keyset = set(body['data'].keys())
                dataframe_optional_data = set(X_predict.keys()).difference(ModelApi.REQUIRED_FIELDS)
                keys_with_data_to_be_added = optional_data_keyset.intersection(dataframe_optional_data)

            results = []
            for index, row in X_predict.iterrows():
                if row['Pollutant'] is not None and math.isnan(row['Pollutant']):
                    continue
                input_instance = {
                    'date_time': index,
                    'longitude': row['Longitude'],
                    'latitude': row['Latitude'],
                    'pollutant': body['pollutant'],
                    'pollution_value': row['Pollutant'],
                    'uncertainty': row['Uncertainty'],
                    'data': {}
                }

                if 'data' in body:
                    for key in keys_with_data_to_be_added:
                        input_instance['data'][key] = row[key]

                result = DatasetsApi.insert_single_instance(input_instance, predicted=True)
                result = DatasetsApi.insert_single_prediction(input_instance)
                results.append(result)
            predictions = ModelApi.__predictions_to_primitive_float(predictions)
            print('failed following: ')
            print(list(filter(lambda x: not x[0], results)))

            return True, predictions

        return False, predictions  # in case that model does not exist

    @staticmethod
    def get_model_params(name):
        """
        Get given model's parameters that are saved in the DB
        :param name: str - name of the model that is saved in the DB
        :return: (None, str) | (dict, None) - str is error message, dict contains model parameters
        """
        model, err = DBManager.get_model_by_name(name)

        if model is None:
            return None, err

        model_params = json.loads(model.model_params)
        # Sometimes it is possible model to give stringified JSON, in that case make it dict
        if 'architecture' in model_params and isinstance(model_params['architecture'], str):
            model_params['architecture'] = json.loads(model_params['architecture'])

        # Do the same for weights
        if 'weights' in model_params and isinstance(model_params['weights'], str):
            model_params['weights'] = json.loads(model_params['weights'])

        model_data = {
            'name': model.name,
            'type': model.type,
            'model_params': model_params,
            'extra_params': json.loads(model.extra_params)
        }

        return model_data, None

    @staticmethod
    def get_model_by_name(name):
        """
        Get a model from database and reproduce it given the parameters saved
        :param name: str - name of the model
        :return: (None, None, str) | (None, dict, str) | (BaseModel, dict, None) - str is error message, dict is model's
        parameters from DB, BaseModel is the instance of the model, might be ConvolutionalNeuralNetwork,
        GaussianProcesses, SparseGaussianProcesses up to date...
        """
        model_record, err = DBManager.get_model_by_name(name)
        if model_record is None:
            return None, None, err

        if model_record.type == 'CNN':
            cnn, err = ConvolutionalNeuralNetwork.new_from_json(model_record.model_params, model_record.extra_params)
            return cnn, model_record, None
        elif model_record.type == 'FullGP':
            full_gp, err = GaussianProcesses.new_from_json(model_record.model_params, model_record.extra_params)
            return full_gp, model_record, None
        elif model_record.type == 'SparseGP':
            sparse_gp, err = SparseGaussianProcesses.new_from_json(model_record.model_params, model_record.extra_params)
            return sparse_gp, model_record, None

        return None, model_record, err

    @staticmethod
    def get_models_by_type(type):
        """
        Get all models that are of given type (CNN, GP, SparseGP)
        :param type: str - 'CNN', 'FullGP', 'SparseGP', other input is invalid
        :return: :return: list | None - for schema data
        """
        if not isinstance(type, str):
            return None, Errors.WRONG_PARAM.value

        models, msg = DBManager.get_models_metadata_by_type(type)
        return models, msg

    @staticmethod
    def train_model(model_name, body):
        """
        Function for further training a model provided that the model already exists in the DB
        :param model_name: str - name of the existing model
        :param body: dict - body of the request
        :return: (True, None) | (False, str) | (False, list)
        """
        print('Getting dataset...')
        model, model_record, err = ModelApi.get_model_by_name(model_name)

        if model is None:
            return False, err

        dataset = DatasetsApi.get_dataset(body, use_dataframe=True)
        if dataset is None:
            return False, Errors.NO_DATA.value

        complete_dataset = dataset[dataset['Pollutant'].notnull()]

        if 'n_instances_trained' in model.stats and 'dataset_stats' in model.stats:
            updated_stats, new_stats = MainTransformer.normalize_with_old_stats(model.stats['n_instances_trained'],
                                                                                model.stats['dataset_stats'],
                                                                                complete_dataset)
            MainTransformer.normalize(complete_dataset, stats=updated_stats, inplace=True)
        else:
            return False, []

        stats = new_stats

        X_train, y_train, _, _, _ = MainTransformer.get_training_and_test_set(complete_dataset,
                                                                              'Pollutant',
                                                                              'Uncertainty',
                                                                              size=1,
                                                                              normalize=False)

        training_dataset_stats = {}
        print('Verifying dataset...')
        if 'dataset_stats' in model.stats:
            training_dataset_stats = model.stats['dataset_stats']
            feature_names = set(training_dataset_stats.keys())
            dataset_features = set(X_train)
            dataset_features.discard('DateTime')

            print('Verifying dataset features')
            if feature_names != dataset_features:
                print('feature names', feature_names, training_dataset_stats, training_dataset_stats.keys())
                print('dataset features', dataset_features)
                if feature_names.intersection(dataset_features) == feature_names:
                    print('Dataset is in the expected shape')
                    print('difference')
                    difference = dataset_features.difference(feature_names)
                    print(difference)
                    MainTransformer.remove_features(X_train, difference)
                else:
                    print(feature_names)
                    print(dataset_features)
                    return False, []
        else:
            return False, []

        print('Starting to train model...')
        model.train(X_train, y_train, stats=stats)
        model_params, extra_params = model.model_to_json()
        result = DBManager.upsert_model(model_name, model_record.type, model_record.resource,
                                        model_params=model_params, extra_params=extra_params)
        print(result)
        return result

        # return False, Errors.NO_SUCH_MODEL_TYPE.value

    @staticmethod
    def get_model_types():
        """
        Get implemented in the core system types of models
        :return: list
        """
        return MODEL_TYPES

    @staticmethod
    def get_all_models():
        """
        Function for getting all models' names and types only from DB
        :return: list | None
        """
        try:
            return DBManager.get_all_models()
        except:
            return None

    @staticmethod
    def __predictions_to_primitive_float(predictions):
        """
        Some output may be np.float which sometimes causes issues, so convert back to primitive values
        :param predictions: np.array | list of np.float
        :return: np.array | list of float
        """
        converted_predictions = []
        while len(predictions) != 0:
            prediction = predictions.pop(0)
            predicted_value = None
            uncertainty = None
            if isinstance(prediction[0], numpy.float32):
                predicted_value = prediction[0].item()

            if isinstance(prediction[1], numpy.float32):
                uncertainty = prediction[1].item()

            converted_prediction = {
                'prediction': predicted_value,
                'uncertainty': uncertainty
            }

            converted_predictions.append(converted_prediction)
        return converted_predictions


