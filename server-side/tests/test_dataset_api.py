import datetime
import json

import pandas
import pytest
import copy

import os
import sys

sys.path.append(os.getcwd())

from api import DatasetsApi
from airpyllution.DB import DBManager as DBM

result, _ = DBM.connect()
DBManager = DBM

testing_data_wrong_input = [
    (None, None, None, None, None, None),
    ({}, None, None, None, None, None),
    ({}, {}, None, None, None, None),
    (None, None, None, None, True, None),
    ({}, None, None, None, True, None),
    ({}, {}, None, None, True, None),
    (None, None, None, None, False, None),
    ({}, None, None, None, False, None),
    ({}, {}, None, None, False, None)
]

testing_data_empty_input = [
    ({}, {}, {}, None, None, list),
    ({}, {}, {}, None, True, pandas.DataFrame),
    ({}, {}, {}, None, False, list),
    ({}, {}, {}, {}, None, list),
    ({}, {}, {}, {}, True, pandas.DataFrame),
    ({}, {}, {}, {}, False, list),
]

helper = lambda x: isinstance(x, str)
helper_no_msg = lambda x: x is None

testing_data_single_measurement = [
    (None, None, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', None, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, None, None, None, (False, helper)),
    (None, 43.23, 44.55, None, None, None, (False, helper)),
    (123, 43.23, 44.55, None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, {}, (True, helper_no_msg)),
]

testing_data_single_prediction = [
    (None, None, None, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', None, None, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, None, None, None, None, (False, helper)),
    (None, 43.23, 44.55, None, None, None, None, (False, helper)),
    (123, 43.23, 44.55, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, None, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, 11.5, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, 11.5, {}, (True, helper_no_msg)),
]

testing_data_single_instance = [
    (None, None, None, None, None, True, None, (False, helper)),
    ('15-05-2018 05:00', None, None, None, None, True, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, None, None, True, None, (False, helper)),
    (None, 43.23, 44.55, None, None, True, None, (False, helper)),
    (123, 43.23, 44.55, None, None, True, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', None, True, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, True, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, True, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, True, {}, (True, helper_no_msg)),
    (None, None, None, None, None, False, None, (False, helper)),
    ('15-05-2018 05:00', None, None, None, None, False, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, None, None, False, None, (False, helper)),
    (None, 43.23, 44.55, None, None, False, None, (False, helper)),
    (123, 43.23, 44.55, None, None, False, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', None, False, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, False, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, False, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, False, {}, (True, helper_no_msg)),
    (None, None, None, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', None, None, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, None, None, None, None, (False, helper)),
    (None, 43.23, 44.55, None, None, None, None, (False, helper)),
    (123, 43.23, 44.55, None, None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', None, None, None, (False, helper)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, None, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, None, None, (True, helper_no_msg)),
    ('15-05-2018 05:00', 43.23, 44.55, 'PM10', 12.2, None, {}, (True, helper_no_msg)),
]

generate_prev_records_data = [
    (None, None, None, None),
    ({}, None, None, None),
    ({}, 0, None, None),
    ({}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00']}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': 42.34}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Latitude': 42.34}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': 42.34, 'Latitude': 42.34}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': 42.34, 'Latitude': 42.34, 'Pollutant': None}, 24, None, 25),
    ({'DateTime': ['25-04-2018 15:00']}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': None}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Latitude': None}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': None, 'Latitude': None}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': 'test'}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Latitude': 'test'}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': 'test', 'Latitude': 'test'}, 24, None, None),
    ({'DateTime': ['25-04-2018 15:00'], 'Longitude': 'test', 'Latitude': 'test', 'Pollutant': 'test'}, 24, None, None),
]

get_dataset_body = {
    "range": {
        "start": "30-09-2018 05:00",
        "end": "01-10-2018 05:00"
    },
    "locations": [[-1.463484, 50.920265], [-1.395778, 50.908140]],
    "pollutant": "PM10",
    "data": {
        "weather": {
            "Temperature": None,
            "Humidity": None,
            "Precipitation": None,
            "WindSpeed": None
        }
    }
}

config_params = {
    "Date": DatasetsApi.DATE_TIME_FORMAT.split(' ')[0],
    "Time": DatasetsApi.DATE_TIME_FORMAT.split(' ')[1],
    "pollutant": {
        "Pollutant": None
    },
    'weather': {}
}

get_dataset_body_no_body = copy.deepcopy(get_dataset_body)
del get_dataset_body_no_body['data']

get_dataset_data = [
    (None, True, None),
    (None, False, None),
    ({'range': None}, True, None),
    ({'locations': None}, True, None),
    ({'pollutant': None}, True, None),
    ({'pollutant': None, 'locations': None, 'range': None}, True, None),
    (get_dataset_body, True, 'DBManager'),
    (get_dataset_body, True, 'DBManager')
]

get_single_instance_body = {
    "name": "cnn",
    "date_time": "10-03-2019 05:00",
    "longitude": -1.395778,
    "latitude": 50.90814,
    "pollutant": "PM10",
    "pollution_value": 5.5
}

get_single_instance_dataset_data = [
    (None, None, None, None),
    ({}, None, None, None),
    ({'date_time': None}, None, None, None),
    ({'longitude': None}, None, None, None),
    ({'latitude': None}, None, None, None),
    ({'date_time': None, 'longitude': None, 'latitude': None}, None, None, None),
    (get_single_instance_body, None, 'DBManager', 'NoComparison'),
    (get_single_instance_body, None, 24, 'Comparison'),
    (get_single_instance_body, None, 12, 'Comparison')
]


# For these tests it is assumed that DBManager works as expected and passes its unit tests, it's not necessary to
# check if result is the same in the database as this is part of DBManager's tests...
# Here tests are more like integration tests
class TestDatasetAPI:
    @pytest.mark.parametrize('range,locations,pollutant,data,use_dataframe,expected', testing_data_wrong_input)
    def test_get_dataset_wrong_input(self, range, locations, pollutant, data, use_dataframe, expected):
        body = {
            'range': range,
            'locations': locations,
            'pollutant': pollutant,
            'data': data
        }
        result = DatasetsApi.get_dataset(body, use_dataframe=use_dataframe)

        assert result is expected

    @pytest.mark.parametrize('range,locations,pollutant,data,use_dataframe,expected', testing_data_empty_input)
    def test_get_dataset_empty_input(self, range, locations, pollutant, data, use_dataframe, expected):
        body = {
            'range': range,
            'locations': locations,
            'pollutant': pollutant,
            'data': data
        }
        result = DatasetsApi.get_dataset(body, use_dataframe=use_dataframe)

        assert isinstance(result, expected)

    @pytest.mark.parametrize('date_time,longitude,latitude,pollutant,pollution_value,data,expected',
                             testing_data_single_measurement)
    def test_insert_single_measurement(self, date_time, longitude, latitude, pollutant, pollution_value,
                                       data, expected):
        body = {
            'date_time': date_time,
            'longitude': longitude,
            'latitude': latitude,
            'pollutant': pollutant,
            'pollution_value': pollution_value,
            'data': data
        }
        result, msg = DatasetsApi.insert_single_measurement(body)

        assert result == expected[0] and expected[1](msg)

    @pytest.mark.parametrize('date_time,longitude,latitude,pollutant,pollution_value,uncertainty,data,expected',
                             testing_data_single_prediction)
    def test_insert_single_prediction(self, date_time, longitude, latitude, pollutant, pollution_value,
                                      uncertainty, data, expected):
        body = {
            'date_time': date_time,
            'longitude': longitude,
            'latitude': latitude,
            'pollutant': pollutant,
            'pollution_value': pollution_value,
            'uncertainty': uncertainty,
            'data': data
        }
        result, msg = DatasetsApi.insert_single_prediction(body)

        assert result == expected[0] and expected[1](msg)

    @pytest.mark.parametrize('date_time,longitude,latitude,pollutant,pollution_value,predicted,data,expected',
                             testing_data_single_instance)
    def test_insert_single_instance(self, date_time, longitude, latitude, pollutant, pollution_value,
                                    predicted, data, expected):
        body = {
            'date_time': date_time,
            'longitude': longitude,
            'latitude': latitude,
            'pollutant': pollutant,
            'pollution_value': pollution_value,
            'data': data
        }
        result, msg = DatasetsApi.insert_single_instance(body, predicted=predicted)

        assert result == expected[0] and expected[1](msg)

    def test_get_coordinates(self):
        result = DatasetsApi.get_coordinates()

        assert isinstance(result, list)

        for x in result:
            assert isinstance(x, list) and len(x) == 2

    def test_get_pollutants(self):
        result = DatasetsApi.get_pollutants()

        assert isinstance(result, list)

        for x in result:
            assert isinstance(x, str)

    @pytest.mark.parametrize('current_instance,n_prev,ready_data,expected', generate_prev_records_data)
    def test_generate_previous_records(self, current_instance, n_prev, ready_data, expected):
        current_instance_copied = copy.deepcopy(current_instance)
        result = DatasetsApi.generate_previous_records(current_instance, n_prev, ready_data=ready_data)

        if expected is None:
            assert result is None
        else:
            print(result)
            assert len(result['DateTime']) == expected

            for x in result['Longitude']:
                assert x == current_instance_copied['Longitude']

            for x in result['Latitude']:
                assert x == current_instance_copied['Latitude']

    @pytest.mark.parametrize('body,use_dataframe,expected', get_dataset_data)
    def test_get_dataset(self, body, use_dataframe, expected):
        if expected is None:
            result = DatasetsApi.get_dataset(body)

            assert result is None
            return

        if expected == 'DBManager':
            print(DBManager)
            db_manager_result, err = DBManager.get_dataset(
                datetime_from=datetime.datetime.strptime(get_dataset_body['range']['start'],
                                                         DatasetsApi.DATE_TIME_FORMAT),
                datetime_to=datetime.datetime.strptime(get_dataset_body['range']['end'],
                                                       DatasetsApi.DATE_TIME_FORMAT),
                longitude=body['locations'][0][0],
                latitude=body['locations'][0][1],
                config=config_params,
                use_dataframe=use_dataframe,
                uncertainty=True)

            db_manager_result_second_loc, err = DBManager.get_dataset(
                datetime_from=datetime.datetime.strptime(get_dataset_body['range']['start'],
                                                         DatasetsApi.DATE_TIME_FORMAT),
                datetime_to=datetime.datetime.strptime(get_dataset_body['range']['end'],
                                                       DatasetsApi.DATE_TIME_FORMAT),
                longitude=body['locations'][1][0],
                latitude=body['locations'][1][1],
                config=config_params,
                use_dataframe=use_dataframe,
                uncertainty=True)

            result = DatasetsApi.get_dataset(body)

            if use_dataframe:
                db_manager_dataset_length = db_manager_result.shape[0] + db_manager_result_second_loc.shape[0]
                assert db_manager_dataset_length == result.shape[0]

    @pytest.mark.parametrize('body,stats,prev,expected', get_single_instance_dataset_data)
    def test_get_single_instance_dataset(self, body, stats, prev, expected):
        if expected is None:
            result = DatasetsApi.get_single_instance_dataset(body, stats, prev)

            assert result is None
            return

        result = DatasetsApi.get_single_instance_dataset(body, stats, prev)

        if expected == 'NoComparison':
            assert result.shape[0] == 1
            return

        assert result.shape[0] == prev + 1
