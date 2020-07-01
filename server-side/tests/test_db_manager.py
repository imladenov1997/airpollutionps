import datetime
import json

import pandas
import pytest
import copy

import os
import sys

sys.path.append(os.getcwd())

config = None
DBManager = None


class TestDBManager:
    def test_connect(self):
        # connect to database
        from airpyllution.DB import DBManager as DBM
        from airpyllution.Utils.ConfigReader import ConfigReader

        is_successful, msg = ConfigReader.open_config()
        result, _ = DBM.connect()
        global DBManager
        global config
        config = ConfigReader.CONFIG
        DBManager = DBM
        assert result

    def test_insert_empty_dataset(self):
        empty_dataset = pandas.DataFrame({})
        global config

        result, _ = DBManager.insert_dataset(empty_dataset, config)
        assert result

    def test_insert_without_config(self):
        empty_dataset = pandas.DataFrame({})

        with pytest.raises(TypeError):
            DBManager.insert_dataset(empty_dataset, None)

    def test_insert_empty_config(self):
        single_instance_dataset = pandas.DataFrame({
            'DateTime': ['01-01-2019 15:00'],
            'Longitude': [42.353],
            'Latitude': [32.543],
            'Pollutant': [5.5]
        })

        result, err_msg = DBManager.insert_dataset(single_instance_dataset, {})

        assert not result and err_msg is not None

    def test_insert_missing_config_features(self):
        single_instance_dataset = pandas.DataFrame({
            'DateTime': ['01-01-2019 15:00'],
            'Longitude': [42.353],
            'Latitude': [32.543],
            'Pollutant': [5.5]
        })

        with pytest.raises(KeyError):
            DBManager.insert_dataset(single_instance_dataset, {'pollutant': {'Pollutant': 'PM10'}})

    def test_insert_with_missing_pollutant(self):
        single_instance_dataset = pandas.DataFrame({
            'DateTime': ['01-01-2019 15:00'],
            'Longitude': [42.353],
            'Latitude': [32.543],
            'Pollutant': [5.5]
        })

        sample_config = {
            'pollutant': {
                "Time": "Time",
                "Longitude": "Longitude",
                "Latitude": "Latitude",
                "Pollutant": "PM10"
            }
        }

        with pytest.raises(KeyError):
            DBManager.insert_dataset(single_instance_dataset, sample_config)

    def test_insert_test_item(self):
        single_instance_dataset = pandas.DataFrame({
            'DateTime': ['01-01-2019 15:00'],
            'Longitude': [42.353],
            'Latitude': [32.543],
            'Pollutant': [5.5]
        })
        global config
        copied = copy.copy(config)
        copied['weather'] = {}

        result, _ = DBManager.insert_dataset(single_instance_dataset, copied)

        assert result

    def test_get_dataset_missing(self):
        result, _ = DBManager.get_dataset()
        assert result is None

        date_from = datetime.datetime.strptime('2018-03-22 10:00', DBManager.DATE_TIME_FORMAT)
        date_to = datetime.datetime.strptime('2018-03-23 10:00', DBManager.DATE_TIME_FORMAT)

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to)
        assert result is None

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, config={})
        assert result is None

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, longitude=50.4, config={})
        assert result is None

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, latitude=50.4, config={})
        assert result is None

        sample_config = {
            "Date": "%d-%m-%Y",
            "Time": "%H:%M"
        }
        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, config=sample_config,
                                          use_dataframe=True)
        assert isinstance(result, pandas.DataFrame)

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, config=sample_config,
                                          use_dataframe=False)
        assert isinstance(result, list)

    def test_insert_get(self):
        global config
        copied = copy.copy(config)
        copied['weather'] = {}

        date_from = datetime.datetime.strptime('2019-03-22 09:00', DBManager.DATE_TIME_FORMAT)
        date_to = datetime.datetime.strptime('2019-03-23 11:00', DBManager.DATE_TIME_FORMAT)

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, config=copied,
                                          use_dataframe=True)

        assert result.shape[0] == 0

        single_instance_dataset = pandas.DataFrame({
            'DateTime': ['22-03-2019 10:00'],
            'Longitude': [42.353],
            'Latitude': [32.543],
            'Pollutant': [5.5]
        })

        result, _ = DBManager.insert_dataset(single_instance_dataset, copied)

        assert result

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, config=copied,
                                          use_dataframe=True)

        assert result.shape[0] == 1

    def test_insert_instance(self):
        date_time = datetime.datetime.strptime('2019-01-22 18:00', DBManager.DATE_TIME_FORMAT)
        longitude = 42.353
        latitude = 32.543
        pollutant = 5.5

        result, _ = DBManager.insert_instance()

        assert not result

        result, _ = DBManager.insert_instance(date_time=date_time)

        assert not result

        result, _ = DBManager.insert_instance(date_time=date_time, longitude=longitude)

        assert not result

        result, msg = DBManager.insert_instance(date_time=date_time, longitude=longitude, latitude=latitude)

        assert result and isinstance(msg, str)

        result, msg = DBManager.insert_instance(date_time=date_time, longitude=longitude, latitude=latitude,
                                                pollutant_name='PM10')

        assert result and isinstance(msg, str)

        result, msg = DBManager.insert_instance(date_time=date_time, longitude=longitude, latitude=latitude,
                                                pollutant_name='PM10', pollution_value=5.0)

        assert result and msg is None

        result, msg = DBManager.insert_instance(date_time=date_time, longitude=longitude, latitude=latitude,
                                                pollutant_name=123, pollution_value=5.0)

        assert result and isinstance(msg, str)

    def test_insert_instance_test_item(self):
        date_time = datetime.datetime.strptime('2019-03-22 19:00', DBManager.DATE_TIME_FORMAT)
        longitude = 42.353
        latitude = 32.543
        pollutant = 5.5
        pollutant_name = 'PM10'
        global config
        date_from = datetime.datetime.strptime('2019-03-22 18:00', DBManager.DATE_TIME_FORMAT)
        date_to = datetime.datetime.strptime('2019-03-23 20:00', DBManager.DATE_TIME_FORMAT)

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, config=config,
                                          use_dataframe=True)

        assert isinstance(result, pandas.DataFrame)

        size = result.shape[0]

        result, _ = DBManager.insert_instance(date_time=date_time, longitude=longitude, latitude=latitude,
                                              pollutant_name=pollutant_name, pollution_value=pollutant, data={},
                                              predicted=True)

        assert result

        result, _ = DBManager.get_dataset(datetime_from=date_from, datetime_to=date_to, config=config,
                                          use_dataframe=True)

        assert isinstance(result, pandas.DataFrame)
        assert result.shape[0] == size + 1

    def test_upsert_model(self):
        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1')

        assert result

        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1', model_params={'test': 1})

        assert not result

        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1', extra_params={'test': 1})

        assert not result

        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1',
                                             model_params=json.dumps({'test': 1}))

        assert result

        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1',
                                             extra_params=json.dumps({'test': 1}))

        assert result

        result, msg = DBManager.get_model_by_name('TestModel1')

        assert result is not None and msg is None

        assert json.dumps({'test': 1}) == result.extra_params

        assert json.dumps({}) == result.model_params

        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1',
                                             extra_params=json.dumps({'test': 1}), model_params=json.dumps({'test': 1}))

        assert result

        result, msg = DBManager.get_model_by_name('TestModel1')

        assert json.dumps({'test': 1}) == result.extra_params
        assert json.dumps({'test': 1}) == result.model_params

        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1', extra_data='cool')

        assert result

        result, msg = DBManager.upsert_model('TestModel1', 'TestType1', 'TestResource1', extra_data='cool')

        assert result

    def test_get_model_by_name(self):
        result, msg = DBManager.get_model_by_name(datetime.datetime.now().strftime(DBManager.DATE_TIME_FORMAT))

        assert result is None and isinstance(msg, str)

        result, msg = DBManager.get_model_by_name('TestModel1')

        assert result is not None and result.name == 'TestModel1'

    def test_get_models_metadata_by_type(self):
        result, msg = DBManager.get_models_metadata_by_type(None)

        assert result is None and isinstance(msg, str)

        result, msg = DBManager.get_models_metadata_by_type('TestType1')

        assert isinstance(result, list)

    def test_get_models_metadata_by_resource(self):
        result, msg = DBManager.get_models_metadata_by_resource(None)

        assert result is None and isinstance(msg, str)

        result, msg = DBManager.get_models_metadata_by_resource(
            datetime.datetime.now().strftime(DBManager.DATE_TIME_FORMAT)
        )

        assert isinstance(result, list) and len(result) == 0 and msg is None

        result, msg = DBManager.get_models_metadata_by_resource('TestResource1')

        assert isinstance(result, list) and len(result) > 0 and msg is None

    def test_insert_prediction(self):
        date_time = datetime.datetime.strptime('2019-01-22 18:00', DBManager.DATE_TIME_FORMAT)
        longitude = 42.353
        latitude = 32.543
        pollutant = 3

        result, _ = DBManager.insert_prediction()

        assert not result

        result, _ = DBManager.insert_prediction(date_time=datetime.datetime.now())

        assert not result

        result, _ = DBManager.insert_prediction(date_time=datetime.datetime.now(), longitude=longitude)

        assert not result

        result, msg = DBManager.insert_prediction(date_time=datetime.datetime.now(), longitude=longitude,
                                                  latitude=latitude)

        assert not result and isinstance(msg, str)

        result, msg = DBManager.insert_prediction(date_time=datetime.datetime.now(), longitude=longitude,
                                                  latitude=latitude, pollutant_name='PM10')

        assert not result and isinstance(msg, str)

        result, msg = DBManager.insert_prediction(date_time=datetime.datetime.now(), longitude=longitude,
                                                  latitude=latitude, pollutant_name='PM10', pollution_value=5.0)

        assert result and msg is None

        result, msg = DBManager.insert_prediction(date_time=datetime.datetime.now(), longitude=longitude,
                                                  latitude=latitude, pollutant_name='PM10', pollution_value=5.0,
                                                  uncertainty=10.0)

        assert result and msg is None

        result, msg = DBManager.insert_prediction(date_time=datetime.datetime.now(), longitude=longitude,
                                                  latitude=latitude, pollutant_name=123, pollution_value=5.0)

        assert not result and isinstance(msg, str)

    def test_insert_prediction_test_item(self):
        start = datetime.datetime.strptime('2019-03-22 18:00', DBManager.DATE_TIME_FORMAT)
        date_time = datetime.datetime.strptime('2019-03-22 19:00', DBManager.DATE_TIME_FORMAT)
        end = datetime.datetime.strptime('2019-03-22 20:00', DBManager.DATE_TIME_FORMAT)
        longitude = 42.353
        latitude = 32.543
        pollutant = 12.2
        pollutant_name = 'PM10'
        global config

        result, _ = DBManager.get_dataset(datetime_from=start, datetime_to=end, config=config,
                                          longitude=longitude, latitude=latitude, use_dataframe=True)

        assert isinstance(result, pandas.DataFrame)

        pollution_value_before = result['Pollutant'].tolist()[0]
        if pollutant == pollution_value_before:
            pollutant += 1.5

        result, msg = DBManager.insert_prediction(date_time=date_time, longitude=longitude, latitude=latitude,
                                                  pollutant_name=pollutant_name, pollution_value=pollutant,
                                                  predicted=True, uncertainty=10.0)

        assert result

        result, _ = DBManager.get_dataset(datetime_from=start, datetime_to=end, config=config,
                                          use_dataframe=True)

        assert isinstance(result, pandas.DataFrame)

        pollution_value_after = result['Pollutant'].tolist()[0]

        assert pollutant == pollution_value_after and pollutant != pollution_value_before
