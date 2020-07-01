import datetime
import json
import numpy as np

import pandas
import pytest
import copy

import os
import sys

sys.path.append(os.getcwd())

from airpyllution import JSONSaver
from airpyllution.Utils.ConfigReader import ConfigReader

ConfigReader.open_config()

config = ConfigReader.CONFIG
DBManager = None

testing_data_one = pandas.DataFrame({
    'DateTime': ['23-01-2019 05:00'],
    'Longitude': [43.29],
    'Latitude': [42.85]
})
testing_data_one.set_index(keys='DateTime', inplace=True)

testing_data_three = pandas.DataFrame({
    'DateTime': ['23-01-2019 05:00', '23-01-2019 05:00'],
    'Longitude': [43.29, 45.54],
    'Latitude': [42.85, 23.43]
})
testing_data_three.set_index(keys='DateTime', inplace=True)

testing_data_four = pandas.DataFrame({
    'DateTime': ['23-01-2019 05:00', '23-01-2019 05:00'],
    'Longitude': [43.29, 45.54],
    'Latitude': [42.85, 23.43]
})
testing_data_four.set_index(keys='DateTime', inplace=True)

predictions_one = np.array([(5.4, None)])
predictions_two = np.array([(5.4, None), (5.9, None), (3.8, None)])
target_one = pandas.Series([[5.3]])
target_two = pandas.Series([[5.3], [5.7], [3.3]])

testing_data = [
    (None, testing_data_one, predictions_one),
    ({}, testing_data_one, predictions_one),
    (config, testing_data_one, predictions_one),
    (config, pandas.DataFrame(), np.array([[]])),
    (config, testing_data_three, predictions_two)
]

testing_data_eval = [
    (None, testing_data_one, predictions_one, target_one, 'rmse', 5.5),
    ({}, testing_data_one, predictions_one, target_one, 'rmse', 5.5),
    (config, testing_data_one, predictions_one, target_one, 'rmse', 5.5),
    (config, pandas.DataFrame(), np.array([[]]), pandas.Series([[]]), 'mae', 6.4),
    (config, testing_data_three, predictions_two, target_two, 'mape', 3.3)
]


class TestDataSavers:
    @pytest.mark.parametrize('given_config,X_test,given_predictions', testing_data)
    def test_save_predictions(self, given_config, X_test, given_predictions):
        if given_config is None or 'pollutant' not in given_config or 'Pollutant' not in given_config['pollutant']:
            with pytest.raises(ValueError):
                saver = JSONSaver(config=given_config)
        elif 'predictionFile' not in given_config:
            with pytest.raises(ValueError):
                saver = JSONSaver(config=given_config)
        else:
            saver = JSONSaver(config=given_config)
            saver.save_predictions(X_test, given_predictions)

    @pytest.mark.parametrize('given_config,X_test,given_predictions,target,metrics,error', testing_data_eval)
    def test_save_evaluations(self, given_config, X_test, given_predictions, target, metrics, error):
        if given_config is None or 'pollutant' not in given_config or 'Pollutant' not in given_config['pollutant']:
            with pytest.raises(ValueError):
                saver = JSONSaver(config=given_config)
        elif 'predictionFile' not in given_config:
            with pytest.raises(ValueError):
                saver = JSONSaver(config=given_config)
        else:
            saver = JSONSaver(config=given_config)
            saver.save_evaluations(X_test, given_predictions, target, metrics, error)
