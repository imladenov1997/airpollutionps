import os
import sys
import numpy as np
from datetime import datetime

import pandas
import pytest

sys.path.append(os.getcwd())

from airpyllution.DataTransformers.MainTransformer import MainTransformer, WrongConfigTypeException
from airpyllution.DataTransformers.TransformerEnum import Transformers
from airpyllution.Utils.ConfigReader import ConfigReader
from airpyllution import DB

ConfigReader.open_config()
config = ConfigReader.CONFIG

db_dataset, err = DB.DBManager.get_dataset(datetime_from=datetime.strptime("01-01-2018 01:00", '%d-%m-%Y %H:%M'),
                                           datetime_to=datetime.strptime("03-08-2018 06:00", '%d-%m-%Y %H:%M'),
                                           # longitude=-1.395778,
                                           # latitude=50.908140,
                                           config=ConfigReader.CONFIG)


class TestMainTransformer:
    added_transformers = [
        None,
        '123',
        Transformers.WEATHER_TRANSFORMER,
        Transformers.POLLUTANT_TRANSFORMER
    ]

    @pytest.mark.parametrize('transformer', added_transformers)
    def test_add_transformer(self, transformer):
        main_transformer = MainTransformer(config=config)
        result = main_transformer.add_transformer(transformer)

        if not isinstance(transformer, Transformers):
            assert not result

        if 'weather' in main_transformer.transformers or 'pollutant' in main_transformer.transformers:
            assert result

        result = main_transformer.add_transformer(transformer)

        assert not result

    configs = [
        None,
        {},
        ConfigReader.CONFIG
    ]

    @pytest.mark.parametrize('given_config', configs)
    def test_config(self, given_config):
        if given_config is None:
            with pytest.raises(WrongConfigTypeException):
                print('test')
                main_transformer = MainTransformer(config=given_config)
                assert True
        else:
            main_transformer = MainTransformer(config=given_config)
            assert True

    added_transformers = [
        (Transformers.WEATHER_TRANSFORMER, Transformers.POLLUTANT_TRANSFORMER),
        (None, Transformers.POLLUTANT_TRANSFORMER),
        (None, None),
        (Transformers.WEATHER_TRANSFORMER, None),
    ]

    @pytest.mark.parametrize('weather_transformer,pollutant_transformer', added_transformers)
    def test_get_transformers(self, weather_transformer, pollutant_transformer):
        main_transformer = MainTransformer(config=config)
        result_weather = main_transformer.add_transformer(weather_transformer)
        result_pollutant = main_transformer.add_transformer(pollutant_transformer)

        if result_weather ^ ('weather' in main_transformer.transformers):
            assert False

        if result_pollutant ^ ('pollutant' in main_transformer.transformers):
            assert False

    def test_init_transformers(self):
        data_transformer = MainTransformer(config=ConfigReader.CONFIG)
        data_transformer.add_transformer(Transformers.WEATHER_TRANSFORMER)
        data_transformer.add_transformer(Transformers.POLLUTANT_TRANSFORMER)
        data_transformer.transform()

    def test_transform(self):
        main_transformer = MainTransformer(config=config)
        result_weather = main_transformer.add_transformer(Transformers.WEATHER_TRANSFORMER)
        result_pollutant = main_transformer.add_transformer(Transformers.POLLUTANT_TRANSFORMER)
        weather_size = len(config['weather'].keys())
        pollutant_size = len(config['pollutant'].keys())

        main_transformer.transform()
        dataset = main_transformer.get_dataset()

        assert weather_size + pollutant_size == dataset.shape[1]

    def test_periodic_f_add_and_remove(self):
        periodic_f = [
            'TimeSin',
            'TimeCos',
            'DateSin',
            'DateCos'
        ]

        copied_dataset = db_dataset.loc[:]
        initial_keys = set(copied_dataset.keys())

        keys = set()
        for func in periodic_f:
            if func in initial_keys:
                break
        else:
            MainTransformer.periodic_f(copied_dataset)
            keys = set(copied_dataset.keys())

        is_successful = False
        for func in periodic_f:
            if func not in keys:
                break
        else:
            is_successful = True
            assert is_successful

        if not is_successful:
            assert is_successful
            return is_successful

        MainTransformer.remove_periodic_f(copied_dataset)

        keys = set(copied_dataset.loc[:])
        for func in periodic_f:
            if func in keys:
                break
        else:
            assert initial_keys == keys
            return initial_keys == keys

        assert False
        return False

    stats = {
        'TestColumn': {
            'mean': 5,
            'std': 2
        },
        'TestColumn2': {
            'mean': 3,
            'std': 4
        }
    }

    @pytest.mark.parametrize('stats', [stats, None])
    def test_normalize_and_unnormalize(self, stats):
        copied_dataset = pandas.DataFrame({
            'TestColumn': [1.0, 2.0, 3.0],
            'TestColumn2': [5.0, 6.0, 7.0]
        })

        unnormalized_dataset = copied_dataset.copy()

        normalized, statistics = MainTransformer.normalize(copied_dataset, inplace=False, stats=stats)

        if stats is None:
            mean_one = np.mean(copied_dataset['TestColumn'])
            std_one = np.std(copied_dataset['TestColumn'])

            mean_two = np.mean(copied_dataset['TestColumn2'])
            std_two = np.std(copied_dataset['TestColumn2'])
        else:
            mean_one = stats['TestColumn']['mean']
            std_one = stats['TestColumn']['std']

            mean_two = stats['TestColumn2']['mean']
            std_two = stats['TestColumn2']['std']

        copied_dataset['TestColumn'] -= mean_one
        copied_dataset['TestColumn2'] -= mean_two

        copied_dataset['TestColumn'] /= std_one
        copied_dataset['TestColumn2'] /= std_two

        manual_stats = {
            'TestColumn': {
                'mean': mean_one,
                'std': std_one
            },
            'TestColumn2': {
                'mean': mean_two,
                'std': std_two
            }
        }

        assert copied_dataset.equals(normalized)
        assert statistics == manual_stats

        MainTransformer.unnormalize(normalized, statistics, inplace=True)

        print(unnormalized_dataset)
        print(normalized)
        assert unnormalized_dataset.equals(normalized)

    @pytest.mark.parametrize('stats', [None, {}])
    def test_unnormalize_without_stats(self, stats):
        copied_dataset = pandas.DataFrame({
            'TestColumn': [1.0, 2.0, 3.0],
            'TestColumn2': [5.0, 6.0, 7.0]
        })

        normalized = MainTransformer.unnormalize(copied_dataset, stats, inplace=False)

        if stats is None:
            assert normalized is None and stats is None
        else:
            assert normalized is not None and stats == {}

    @pytest.mark.parametrize('stats', [None, {}])
    def test_normalize_unnormalize_without_dataset(self, stats):
        result_normalized, _ = MainTransformer.normalize(None, {}, inplace=False)
        result_unnormalize = MainTransformer.unnormalize(None, {}, inplace=False)

        assert result_normalized is None and result_unnormalize is None

    copied_dataset_one = pandas.DataFrame({
        'TestColumn': [1.0, 2.0],
        'Pollutant': [5.0, 6.0]
    })

    copied_dataset_two = pandas.DataFrame({
        'TestColumn': [1.0, 2.0, 3.0, 4.0],
        'Pollutant': [5.0, 6.0, 7.0, 10.0],
    })

    copied_dataset_three = pandas.DataFrame({
        'TestColumn': [1.0, 2.0, 1.0, 2.0, 1.0],
        'Pollutant': [5.0, 6.0, 1.0, 2.0, 1.0]
    })

    copied_dataset_four = pandas.DataFrame({
        'TestColumn': [],
        'Pollutant': [],
    })

    testing_data = [
        (copied_dataset_one, 0.5, (1, 1)),
        (copied_dataset_two, 0.75, (3, 1)),
        (copied_dataset_three, 0.5, (2, 3)),
        (copied_dataset_four, 1, (0, 0))
    ]

    @pytest.mark.parametrize('df,portion,expected', testing_data)
    def test_get_training_and_test_set(self, df, portion, expected):
        X_train, y_train, X_test, y_test, _ = MainTransformer.get_training_and_test_set(df,
                                                                                            'Pollutant',
                                                                                            None,
                                                                                            size=portion,
                                                                                            normalize=False)
        assert X_train.shape[0] == expected[0] and X_test.shape[0] == expected[1]
        assert X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]
    #
    # def test_init_transformer(self):
    #     pass

    copied_dataset_four = pandas.DataFrame({
        'TestColumn': []
    })

    testing_data = [
        (copied_dataset_one, 'Pollutant', {'TestColumn'}),
        (copied_dataset_two, ['Pollutant', 'TestColumn'], set()),
        (copied_dataset_three, [], {'Pollutant', 'TestColumn'}),
        (copied_dataset_four, 'Pollutant', {'TestColumn'})
    ]

    @pytest.mark.parametrize('df,removed_elements,left_elements', testing_data)
    def test_remove_features(self, df, removed_elements, left_elements):
        dataset = MainTransformer.remove_features(df, removed_elements)
        assert set(dataset.keys()) == left_elements
