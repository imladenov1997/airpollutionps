import copy
import json
import sys
import os
from datetime import datetime

import pytest
import numpy as np

sys.path.append(os.getcwd())  # workaround to find tests, something weird from conda

from airpyllution import DB
from airpyllution.DataTransformers.MainTransformer import MainTransformer
from airpyllution.Models.CNN import ConvolutionalNeuralNetwork, NotEnoughInstancesError, WrongNumberOfFeatures
from airpyllution.Utils.ConfigReader import ConfigReader
from airpyllution.Utils.Errors import Errors
from airpyllution.Models.FullGP import GaussianProcesses
from airpyllution import Metrics
from airpyllution import SparseGaussianProcesses
from airpyllution.DataTransformers.TransformerEnum import Transformers

ConfigReader.open_config()

cnn = None
full_gp = None
sparse_gp = None
dataset, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("01-01-2018 01:00", '%d-%m-%Y %H:%M'),
                                        datetime_to=datetime.strptime("03-01-2018 06:00", '%d-%m-%Y %H:%M'),
                                        # longitude=-1.395778,
                                        # latitude=50.908140,
                                        config=ConfigReader.CONFIG)

copied = copy.copy(ConfigReader.CONFIG)
copied['loadedModel'] = 'cnn_test'


class TestCNN:
    global copied
    MODEL_PARAMS = None
    EXTRA_PARAMS = None

    def test_train(self):
        seq_length = 24
        cnn = ConvolutionalNeuralNetwork(seq_length)

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.85,
                                                                                          normalize=True)

        assert not cnn.is_built
        assert cnn.n_features is None
        assert cnn.seq_length == seq_length
        assert cnn.stats['dataset_stats'] != stats

        cnn.train(X_train_set, y_train_set, stats)

        assert cnn.is_built
        assert cnn.n_features == X_train_set.shape[1]
        assert cnn.seq_length == seq_length
        assert cnn.stats['dataset_stats'] == stats

    def test_train_not_enough_instances(self):
        cnn = ConvolutionalNeuralNetwork(24)

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.01,
                                                                                          normalize=True)

        with pytest.raises(NotEnoughInstancesError):
            cnn.train(X_train_set, y_train_set, stats=stats)

    def test_predict_not_enough_instances(self):
        global cnn
        cnn = ConvolutionalNeuralNetwork(24)

        X_train_set, y_train_set, X_test, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                               'Pollutant',
                                                                                               'Uncertainty',
                                                                                               size=0.95,
                                                                                               normalize=True)

        cnn.train(X_train_set, y_train_set, stats=stats)

        predictions = cnn.predict(X_test=X_test, uncertainty=True)

        n_none_predictions = len(list(filter(lambda x: x[0] is None and x[1] is None, predictions)))

        assert X_test.shape[0] == n_none_predictions

    def test_train_and_test(self):
        cnn = ConvolutionalNeuralNetwork(24)

        # just testing, don't care about overfitting
        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=1,
                                                                                          normalize=True)

        cnn.train(X_train_set, y_train_set, stats=stats)
        predictions = cnn.predict(X_train_set)

        assert len(predictions) == X_train_set.shape[0]

    def test_train_and_test_no_uncertainty(self):
        cnn = ConvolutionalNeuralNetwork(24)

        # just testing, don't care about overfitting
        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=1,
                                                                                          normalize=True)

        cnn.train(X_train_set, y_train_set, stats=stats)
        predictions = cnn.predict(X_train_set, uncertainty=False)

        assert len(predictions) == X_train_set.shape[0]

    def test_train_and_test_no_uncertainty_not_enough_instances(self):
        cnn = ConvolutionalNeuralNetwork(24)

        # just testing, don't care about overfitting
        X_train_set, y_train_set, X_test, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                               'Pollutant',
                                                                                               'Uncertainty',
                                                                                               size=0.95,
                                                                                               normalize=True)

        cnn.train(X_train_set, y_train_set, stats=stats)
        predictions = cnn.predict(X_test, uncertainty=False)

        n_none_predictions = len(list(filter(lambda x: x[0] is None and x[1] is None, predictions)))

        assert n_none_predictions == len(X_test)

    def test_eval(self):
        cnn = ConvolutionalNeuralNetwork(24)

        dataset, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("01-01-2018 01:00", '%d-%m-%Y %H:%M'),
                                                datetime_to=datetime.strptime("06-01-2018 06:00", '%d-%m-%Y %H:%M'),
                                                # longitude=-1.395778,
                                                # latitude=50.908140,
                                                config=ConfigReader.CONFIG)

        # just testing, don't care about overfitting
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.8,
                                                                                                    normalize=True)

        cnn.train(X_train_set, y_train_set, stats=stats)
        result, predictions, y_test_set = cnn.eval(X_test, y_test)

        predictions_size = len(predictions)

        assert predictions_size == len(X_test)

    def test_eval_not_enough(self):
        cnn = ConvolutionalNeuralNetwork(24)

        dataset, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("01-01-2018 01:00", '%d-%m-%Y %H:%M'),
                                                datetime_to=datetime.strptime("03-01-2018 06:00", '%d-%m-%Y %H:%M'),
                                                # longitude=-1.395778,
                                                # latitude=50.908140,
                                                config=ConfigReader.CONFIG)

        # just testing, don't care about overfitting
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.8,
                                                                                                    normalize=True)

        cnn.train(X_train_set, y_train_set, stats=stats)
        result, predictions, y_test_set = cnn.eval(X_test, y_test)

        predictions_size = len(predictions)

        assert predictions_size == len(X_test)

    def test_save(self):
        global cnn
        config = ConfigReader.CONFIG
        copied = copy.copy(config)
        copied['persistence'] = {
            'modelName': 'cnn_test'
        }
        result, msg = cnn.save_model(config=copied)

        assert result and msg is None

    def test_save_wrong_format(self):
        global cnn
        config = ConfigReader.CONFIG
        copied = copy.copy(config)
        copied['persistence'] = copied['persistence'] = 'cnn_test'
        result, msg = cnn.save_model(config=copied)

        assert not result and msg == Errors.NO_MODEL_DATA.value

    def test_save_no_config(self):
        global cnn
        result, msg = cnn.save_model(config=None)

        assert not result and msg == Errors.WRONG_CONFIG.value

    def test_save_empty_config(self):
        global cnn
        result, msg = cnn.save_model(config={})

        assert not result and msg == Errors.MODEL_NO_NAME.value

    def test_load_saved_model(self):
        global cnn
        config = ConfigReader.CONFIG
        copied = copy.copy(config)
        copied['loadedModel'] = {
            'modelName': 'cnn_test'
        }

        result, msg = cnn.load_model(config=copied)

        assert result and msg is None

    def test_load_saved_model_no_config(self):
        global cnn
        result, msg = cnn.load_model(config=None)

        assert not result and msg == Errors.WRONG_CONFIG.value

    configs = [
        {},
        copied
    ]

    @pytest.mark.parametrize('given_config', configs)
    def test_load_saved_model_wrong_format(self, given_config):
        global cnn
        result, msg = cnn.load_model(config=given_config)

        assert not result and msg == Errors.NO_SUCH_MODEL.value

    def test_model_to_json(self):
        global cnn
        model_params, extra_params = cnn.model_to_json()
        model_params_dict = json.loads(model_params)
        extra_params_dict = json.loads(extra_params)
        are_model_params_set = 'architecture' in model_params_dict and \
                               model_params_dict['architecture'] == cnn.model.to_json() and \
                               'weights' in model_params_dict

        are_extra_params_set = 'sequence_length' in extra_params_dict and \
                               extra_params_dict['sequence_length'] == cnn.seq_length and \
                               'n_features' in extra_params_dict and \
                               extra_params_dict['n_features'] == cnn.n_features and \
                               'stats' in extra_params_dict and 'dataset_stats' in extra_params_dict['stats'] and \
                               len(extra_params_dict['stats']['dataset_stats'].keys()) == \
                               extra_params_dict['n_features']

        assert are_model_params_set and are_extra_params_set

    def test_load_from_json(self):
        global cnn
        model_params, extra_params = cnn.model_to_json()

        model_params_dict = json.loads(model_params)
        extra_params_dict = json.loads(extra_params)

        loaded_cnn, error_msg = ConvolutionalNeuralNetwork.new_from_json(model_params_dict, extra_params_dict)

        assert error_msg is None

        assert loaded_cnn.n_features == extra_params_dict['n_features']
        assert loaded_cnn.stats == extra_params_dict['stats']

    def test_update_stats(self):
        seq_length = 24
        updated_cnn = ConvolutionalNeuralNetwork(seq_length)

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.85,
                                                                                          normalize=True)

        updated_cnn.train(X_train_set, y_train_set, stats)

        instances = updated_cnn.stats['n_instances_trained']
        dataset_stats = updated_cnn.stats['dataset_stats']

        assert X_train_set.shape[0] == instances
        assert stats == dataset_stats

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.5,
                                                                                          normalize=True)

        updated_cnn.train(X_train_set, y_train_set, stats)

        assert X_train_set.shape[0] + instances == updated_cnn.stats['n_instances_trained']
        assert len(cnn.stats['dataset_stats'].keys()) == len(stats.keys()) == len(dataset_stats.keys())

        missing_data = X_train_set.drop(axis=1, columns='Temperature', inplace=False, errors='ignore')

        with pytest.raises(WrongNumberOfFeatures):
            updated_cnn.train(missing_data, y_train_set, stats)


dataset_gp, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("01-01-2018 01:00", '%d-%m-%Y %H:%M'),
                                           datetime_to=datetime.strptime("03-07-2018 06:00", '%d-%m-%Y %H:%M'),
                                           # longitude=-1.395778,
                                           # latitude=50.908140,
                                           config=ConfigReader.CONFIG)

dataset_gp_two, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("01-01-2018 01:00", '%d-%m-%Y %H:%M'),
                                               datetime_to=datetime.strptime("03-04-2018 06:00", '%d-%m-%Y %H:%M'),
                                               # longitude=-1.395778,
                                               # latitude=50.908140,
                                               config=ConfigReader.CONFIG)

dataset_gp_three, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("04-05-2018 01:00", '%d-%m-%Y %H:%M'),
                                                 datetime_to=datetime.strptime("30-09-2018 06:00", '%d-%m-%Y %H:%M'),
                                                 # longitude=-1.395778,
                                                 # latitude=50.908140,
                                                 config=ConfigReader.CONFIG)

dataset_gp_four, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("04-05-2018 01:00", '%d-%m-%Y %H:%M'),
                                                datetime_to=datetime.strptime("06-07-2018 06:00", '%d-%m-%Y %H:%M'),
                                                # longitude=-1.395778,
                                                # latitude=50.908140,
                                                config=ConfigReader.CONFIG)


class TestGP:
    @pytest.mark.parametrize('uncertainty', [True, False, None])
    def test_train_and_test(self, uncertainty):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset_gp,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = GaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        assert gp.stats['n_instances_trained'] == X_train_set.shape[0]
        assert gp.stats['dataset_stats'] == stats

        predictions = gp.predict(X_test, uncertainty=uncertainty)

        assert len(predictions) == X_test.shape[0]

        if uncertainty:
            values_without_uncertainty = list(filter(lambda x: len(x) != 2, predictions))
            assert len(values_without_uncertainty) == 0

        if not isinstance(uncertainty, bool):
            assert len(list(filter(lambda x: not isinstance(x, tuple), predictions))) == X_test.shape[0]

    @pytest.mark.parametrize('given_dataset', [dataset_gp, dataset_gp_two, dataset_gp_three, dataset_gp_four])
    def test_train_and_test_various_datasets(self, given_dataset):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(given_dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = GaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        assert gp.stats['n_instances_trained'] == X_train_set.shape[0]
        assert gp.stats['dataset_stats'] == stats

        predictions = gp.predict(X_test, uncertainty=True)

        assert len(predictions) == X_test.shape[0]

    datasets = [
        (dataset_gp, dataset_gp_two),
        (dataset_gp, dataset_gp_three),
        (dataset_gp, dataset_gp_four),
        (dataset_gp_two, dataset_gp_three),
        (dataset_gp_four, dataset_gp_three),
    ]

    @pytest.mark.parametrize('dataset_one,dataset_two', datasets)
    def test_retrain(self, dataset_one, dataset_two):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset_one,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.6,
                                                                                                    normalize=True)

        gp = GaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        instances = gp.stats['n_instances_trained']
        model_stats = gp.stats['dataset_stats']

        assert instances == X_train_set.shape[0]
        assert model_stats == stats

        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset_two,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp.train(X_train_set, y_train_set, stats)

        assert instances + X_train_set.shape[0] == gp.stats['n_instances_trained']
        assert model_stats != gp.stats['dataset_stats'] != stats

    @pytest.mark.parametrize('error_func', [Metrics.rmse, Metrics.mae, Metrics.mape, Metrics.sse, None])
    def test_eval(self, error_func):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = GaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)
        result, predictions, y_test_set = gp.eval(X_test, y_test, error_func=error_func)

        predictions_size = len(predictions)

        assert predictions_size == len(X_test)

    copied_right = copy.copy(ConfigReader.CONFIG)
    copied_right['persistence'] = {
        'modelName': 'gp_full'
    }
    copied_right['loadedModel'] = {
        'modelName': 'gp_full'
    }
    copied_wrong = copy.copy(ConfigReader.CONFIG)
    copied_wrong['persistence'] = 'gp_full'
    copied_wrong['loadedModel'] = 'gp_full'

    configs = [
        None,
        {},
        copied_wrong
    ]

    @pytest.mark.parametrize('config', configs)
    def test_save_wrong(self, config):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = GaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        result, msg = gp.save_model(config)

        assert not result and isinstance(msg, str)

    @pytest.mark.parametrize('config', [copied_right])
    def test_save(self, config):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = GaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        result, msg = gp.save_model(config)
        global full_gp
        full_gp = gp

        assert result and msg is None

    def test_load_saved_model(self):
        global full_gp
        copied_config = copy.copy(ConfigReader.CONFIG)
        copied_config['loadedModel'] = {
            'modelName': 'gp_full'
        }

        loaded = GaussianProcesses()
        result, msg = loaded.load_model(copied_config)

        assert result and msg is None

        assert full_gp.stats == loaded.stats
        assert full_gp.kernel.to_dict() == loaded.kernel.to_dict()
        assert full_gp.model.param_array.tolist() == loaded.model.param_array.tolist()

    def test_model_to_json_load_from_json(self):
        global full_gp
        model_params, extra_params = full_gp.model_to_json()
        model_params_dict = json.loads(model_params)
        extra_params_dict = json.loads(extra_params)

        assert model_params_dict['data']['kernel'] == full_gp.kernel.to_dict()
        assert model_params_dict['data']['params'] == full_gp.model.param_array.tolist()

        loaded_gp, msg = GaussianProcesses.new_from_json(model_params_dict, extra_params_dict)

        assert msg is None
        assert full_gp.stats == loaded_gp.stats
        assert full_gp.kernel.to_dict() == loaded_gp.kernel.to_dict()
        assert full_gp.model.param_array.tolist() == loaded_gp.model.param_array.tolist()

    def test_update_stats(self):
        full_gp = GaussianProcesses()

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.85,
                                                                                          normalize=True)

        full_gp.train(X_train_set, y_train_set, stats=stats)

        instances = full_gp.stats['n_instances_trained']
        dataset_stats = full_gp.stats['dataset_stats']

        assert X_train_set.shape[0] == instances
        assert stats == dataset_stats

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.5,
                                                                                          normalize=True)

        full_gp.train(X_train_set, y_train_set, stats)

        assert X_train_set.shape[0] + instances == full_gp.stats['n_instances_trained']
        assert len(full_gp.stats['dataset_stats'].keys()) == len(stats.keys()) == len(dataset_stats.keys())

        missing_data = X_train_set.drop(axis=1, columns='Temperature', inplace=False, errors='ignore')

        with pytest.raises(WrongNumberOfFeatures):
            full_gp.train(missing_data, y_train_set, stats)


class TestSparseGP:
    @pytest.mark.parametrize('uncertainty', [True, False, None])
    def test_train_and_test(self, uncertainty):
        global dataset

        data_transformer = MainTransformer(config=ConfigReader.CONFIG)
        data_transformer.add_transformer(Transformers.WEATHER_TRANSFORMER)
        data_transformer.add_transformer(Transformers.POLLUTANT_TRANSFORMER)
        data_transformer.transform()
        dataset = data_transformer.get_dataset()

        complete_dataset = dataset.dropna(inplace=False)
        MainTransformer.periodic_f(complete_dataset)

        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.8,
                                                                                                    normalize=True)

        print(np.array(X_train_set))
        print(y_train_set)

        # X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(complete_dataset,
        #                                                                                             'Pollutant',
        #                                                                                             'Uncertainty',
        #                                                                                             size=0.8,
        #                                                                                             normalize=True)
        #
        # print(np.array(X_train_set))
        # print(X_train_set)
        # print(y_train_set)

        gp = SparseGaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        assert gp.stats['n_instances_trained'] == X_train_set.shape[0]
        assert gp.stats['dataset_stats'] == stats

        predictions = gp.predict(X_test, uncertainty=uncertainty)

        assert len(predictions) == X_test.shape[0]

        if uncertainty:
            values_without_uncertainty = list(filter(lambda x: len(x) != 2, predictions))
            assert len(values_without_uncertainty) == 0

        if not isinstance(uncertainty, bool):
            assert len(list(filter(lambda x: not isinstance(x, tuple), predictions))) == X_test.shape[0]

    def test_retrain(self):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = SparseGaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        instances = gp.stats['n_instances_trained']
        model_stats = gp.stats['dataset_stats']

        assert instances == X_train_set.shape[0]
        assert model_stats == stats

        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=1,
                                                                                                    normalize=True)

        gp.train(X_train_set, y_train_set, stats)

        assert instances + X_train_set.shape[0] == gp.stats['n_instances_trained']
        assert model_stats != gp.stats['dataset_stats'] != stats

    @pytest.mark.parametrize('error_func', [Metrics.rmse, Metrics.mae, Metrics.mape, Metrics.sse, None])
    def test_eval(self, error_func):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = SparseGaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)
        result, predictions, y_test_set = gp.eval(X_test, y_test, error_func=error_func)

        predictions_size = len(predictions)

        assert predictions_size == len(X_test)

    copied_right = copy.copy(ConfigReader.CONFIG)
    copied_right['persistence'] = {
        'modelName': 'gp_sparse'
    }
    copied_right['loadedModel'] = {
        'modelName': 'gp_sparse'
    }
    copied_wrong = copy.copy(ConfigReader.CONFIG)
    copied_wrong['persistence'] = 'gp_sparse'
    copied_wrong['loadedModel'] = 'gp_sparse'

    configs = [
        None,
        {},
        copied_wrong
    ]

    @pytest.mark.parametrize('config', configs)
    def test_save_wrong(self, config):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = SparseGaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        result, msg = gp.save_model(config)

        assert not result and isinstance(msg, str)

    @pytest.mark.parametrize('config', [copied_right])
    def test_save(self, config):
        X_train_set, y_train_set, X_test, y_test, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                                    'Pollutant',
                                                                                                    'Uncertainty',
                                                                                                    size=0.5,
                                                                                                    normalize=True)

        gp = SparseGaussianProcesses()
        gp.train(X_train_set, y_train_set, stats=stats)

        result, msg = gp.save_model(config)
        global sparse_gp
        sparse_gp = gp

        assert result and msg is None

    def test_load_saved_model(self):
        global sparse_gp
        copied_config = copy.copy(ConfigReader.CONFIG)
        copied_config['loadedModel'] = {
            'modelName': 'gp_sparse'
        }

        loaded = SparseGaussianProcesses()
        result, msg = loaded.load_model(copied_config)

        assert result and msg is None

        assert sparse_gp.stats == loaded.stats
        assert sparse_gp.kernel.to_dict() == loaded.kernel.to_dict()
        assert sparse_gp.model.param_array.tolist() == loaded.model.param_array.tolist()

    def test_model_to_json_load_from_json(self):
        global sparse_gp
        model_params, extra_params = sparse_gp.model_to_json()
        model_params_dict = json.loads(model_params)
        extra_params_dict = json.loads(extra_params)

        assert model_params_dict['data']['kernel'] == sparse_gp.kernel.to_dict()
        assert model_params_dict['data']['params'] == sparse_gp.model.param_array.tolist()

        loaded_gp, msg = SparseGaussianProcesses.new_from_json(model_params_dict, extra_params_dict)

        assert msg is None
        assert sparse_gp.stats == loaded_gp.stats
        assert sparse_gp.kernel.to_dict() == loaded_gp.kernel.to_dict()
        assert sparse_gp.model.param_array.tolist() == loaded_gp.model.param_array.tolist()

    def test_update_stats(self):
        sparse_gp = SparseGaussianProcesses()

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.85,
                                                                                          normalize=True)

        sparse_gp.train(X_train_set, y_train_set, stats=stats)

        instances = sparse_gp.stats['n_instances_trained']
        dataset_stats = sparse_gp.stats['dataset_stats']

        assert X_train_set.shape[0] == instances
        assert stats == dataset_stats

        X_train_set, y_train_set, _, _, stats = MainTransformer.get_training_and_test_set(dataset,
                                                                                          'Pollutant',
                                                                                          'Uncertainty',
                                                                                          size=0.5,
                                                                                          normalize=True)

        sparse_gp.train(X_train_set, y_train_set, stats)

        assert X_train_set.shape[0] + instances == sparse_gp.stats['n_instances_trained']
        assert len(sparse_gp.stats['dataset_stats'].keys()) == len(stats.keys()) == len(dataset_stats.keys())

        missing_data = X_train_set.drop(axis=1, columns='Temperature', inplace=False, errors='ignore')

        with pytest.raises(WrongNumberOfFeatures):
            sparse_gp.train(missing_data, y_train_set, stats)
