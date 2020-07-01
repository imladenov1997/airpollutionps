import datetime
import json

import pandas
import pytest
import copy

import os
import sys

from peewee import Model

sys.path.append(os.getcwd())

from airpyllution import ConvolutionalNeuralNetwork
from airpyllution import SparseGaussianProcesses
from airpyllution.Models.FullGP import GaussianProcesses
from api import ModelApi
from airpyllution.DB import DBManager as DBM

result, _ = DBM.connect()
DBManager = DBM

create_model_body = {
    "type": "FullGP",
    "range": {
        "start": "01-01-2018 05:00",
        "end": "10-01-2018 05:00"
    },
    "locations": [[-1.463484, 50.920265], [-1.395778, 50.908140]],
    "pollutant": "PM10"
}

create_model_body_missing_range = copy.copy(create_model_body)
create_model_body_missing_type = copy.copy(create_model_body)
del create_model_body_missing_range['range']
del create_model_body_missing_type['type']

create_model_data = [
    ('Test1', 'CNN', create_model_body, True),
    ('Test2', 'FullGP', create_model_body, True),
    # ('Test3', 'SparseGP', create_model_body, True),
    ('Test4', 'Test', create_model_body, False),
    # ('Test5', 'CNN', create_model_body_missing_range, False),
    ('Test6', 'CNN', create_model_body_missing_type, False)
]

make_predictions_body_test = {
    "name": "Test1",
    "range": {
        "start": "01-01-2018 00:00",
        "end": "1-11-2018 05:00"
    },
    "locations": [[-1.463484, 50.920265], [-1.395778, 50.908140]],
    "pollutant": "PM10",
    "data": {

    },
    "uncertainty": True
}

make_predictions_body_test_range = {
    "name": "Test1",
    "range": {
        "start": "01-05-2018 00:00",
        "end": "1-09-2018 05:00"
    },
    "locations": [[-1.463484, 50.920265], [-1.395778, 50.908140]],
    "pollutant": "PM10",
    "data": {

    },
    "uncertainty": True
}

make_predictions_body_test_location = {
    "name": "Test1",
    "range": {
        "start": "01-05-2018 00:00",
        "end": "1-09-2018 05:00"
    },
    "locations": [[-1.395778, 50.908140]],
    "pollutant": "PM10",
    "data": {

    },
    "uncertainty": True
}

make_predictions_body_missing_range = copy.copy(make_predictions_body_test)
del make_predictions_body_missing_range['range']
make_predictions_body_missing_locations = copy.copy(make_predictions_body_test)
del make_predictions_body_missing_locations['locations']
make_predictions_body_missing_pollutant = copy.copy(make_predictions_body_test)
del make_predictions_body_missing_pollutant['pollutant']

make_predictions_body_wrong_range = copy.copy(make_predictions_body_test)
make_predictions_body_wrong_range['range'] = None
make_predictions_body_wrong_locations = copy.copy(make_predictions_body_test)
make_predictions_body_wrong_locations['range'] = None
make_predictions_body_wrong_pollutant = copy.copy(make_predictions_body_test)
make_predictions_body_wrong_pollutant['range'] = None

predict_data = [
    ('Test1', make_predictions_body_test, True),
    ('NoModel', make_predictions_body_test, False),
    ('Test1', make_predictions_body_test_range, True),
    ('Test1', make_predictions_body_test_location, True),
    ('Test1', {}, False),
    ('Test1', make_predictions_body_missing_range, False),
    ('Test1', make_predictions_body_missing_locations, False),
    ('Test1', make_predictions_body_missing_pollutant, False),
    ('Test1', make_predictions_body_wrong_range, False),
    ('Test1', make_predictions_body_wrong_range, False),
    ('Test1', make_predictions_body_wrong_range, False),
    ('Test2', make_predictions_body_test, True),
    ('Test2', make_predictions_body_test_range, True),
    ('Test2', make_predictions_body_test_location, True),
    ('Test2', {}, False),
    ('Test2', make_predictions_body_missing_range, False),
    ('Test2', make_predictions_body_missing_locations, False),
    ('Test2', make_predictions_body_missing_pollutant, False),
    ('Test2', make_predictions_body_wrong_range, False),
    ('Test2', make_predictions_body_wrong_range, False),
    ('Test2', make_predictions_body_wrong_range, False)
]

predict_instance_body_test_one = {
    "name": "Test1",
    "date_time": "10-03-2019 05:00",
    "longitude": -1.395778,
    "latitude": 50.90814,
    "pollutant": "PM10"
}

predict_instance_body_test_two = {
    "name": "Test1",
    "date_time": "10-03-2018 05:00",
    "longitude": -1.395778,
    "latitude": 50.90814,
    "pollutant": "PM2.5"
}

predict_instance_body_test_three = {
    "name": "Test1",
    "date_time": "10-04-2018 05:00",
    "longitude": -1.392778,
    "latitude": 50.90834,
    "pollutant": "PM2.5"
}

predict_instance_body_test_four = copy.copy(predict_instance_body_test_one)
del predict_instance_body_test_four['name']
predict_instance_body_test_five = copy.copy(predict_instance_body_test_one)
del predict_instance_body_test_five['date_time']
predict_instance_body_test_six = copy.copy(predict_instance_body_test_one)
del predict_instance_body_test_six['longitude']
predict_instance_body_test_seven = copy.copy(predict_instance_body_test_one)
del predict_instance_body_test_seven['latitude']
predict_instance_body_test_seven = copy.copy(predict_instance_body_test_one)
del predict_instance_body_test_seven['pollutant']

predict_instance_data = [
    (None, False),
    ({}, False),
    (predict_instance_body_test_one, True),
    (predict_instance_body_test_two, True),
    (predict_instance_body_test_three, True),
    (predict_instance_body_test_four, False),
    (predict_instance_body_test_five, False),
    (predict_instance_body_test_six, False),
    (predict_instance_body_test_seven, False)
]

model_params_data = [
    ('Test1', 'Params'),
    ('NoModel', None)
]

model_by_name_data = [
    ('Test1', 'CNN'),
    ('Test2', 'FullGP'),
    # ('Test3', 'SparseGP'),
    ('NoModel', None)
]

model_type_data = [
    (123, SparseGaussianProcesses),
    (None, SparseGaussianProcesses),
    ('CNN', ConvolutionalNeuralNetwork),
    ('FullGP', GaussianProcesses),
    ('SparseGP', SparseGaussianProcesses),
    ('NoType', None)
]

train_model_body = {
    "range": {
        "start": "12-01-2018 05:00",
        "end": "15-01-2018 05:00"
    },
    "locations": [[-1.463484, 50.920265], [-1.395778, 50.908140]],
    "pollutant": "PM10"
}

train_model_body_wrong_location = copy.copy(train_model_body)
train_model_body_wrong_location['locations'] = None
train_model_body_wrong_range = copy.copy(train_model_body)
train_model_body_wrong_range['range'] = None
train_model_body_wrong_pollutant = copy.copy(train_model_body)
train_model_body_wrong_pollutant['pollutant'] = None

train_model_body_missing_location = copy.copy(train_model_body)
del train_model_body_missing_location['locations']
train_model_body_missing_range = copy.copy(train_model_body)
del train_model_body_missing_range['range']
train_model_body_missing_pollutant = copy.copy(train_model_body)
del train_model_body_missing_pollutant['pollutant']

train_model_data = [
    ('Test1', train_model_body, True),
    ('Test2', train_model_body, True),
    ('Test1', train_model_body_wrong_location, False),
    ('Test1', train_model_body_wrong_range, False),
    ('Test1', train_model_body_wrong_pollutant, False),
    ('Test1', train_model_body_missing_location, False),
    ('Test1', train_model_body_missing_range, False),
    ('Test1', train_model_body_missing_pollutant, False),
    ('Test2', train_model_body_wrong_location, False),
    ('Test2', train_model_body_wrong_range, False),
    ('Test2', train_model_body_wrong_pollutant, False),
    ('Test2', train_model_body_missing_location, False),
    ('Test2', train_model_body_missing_range, False),
    ('Test2', train_model_body_missing_pollutant, False)
]


class TestModelAPI:
    @pytest.mark.parametrize('name,type,body,expected', create_model_data)
    def test_create_model(self, name, type, body, expected):
        print(body)
        if 'type' in body:
            body['type'] = type

            result, err = ModelApi.create_model(name, body)
            assert result == expected
        else:
            result, err = ModelApi.create_model(name, body)
            assert result == expected

    @pytest.mark.parametrize('name,body,expected', predict_data)
    def test_make_predictions(self, name, body, expected):
        body['name'] = name
        result, predictions = ModelApi.make_predictions(body, overwrite=False)
        assert result == expected

    @pytest.mark.parametrize('body,expected', predict_instance_data)
    def test_make_single_prediction(self, body, expected):
        result, predictions = ModelApi.make_single_prediction(body)
        assert result == expected

    @pytest.mark.parametrize('name,option', model_params_data)
    def test_get_model_params(self, name, option):
        if option == 'Params':
            model, err = ModelApi.get_model_params(name)
            assert err is None and isinstance(model, dict)
        elif option is None:
            model, err = ModelApi.get_model_params(name)
            assert model is None and isinstance(err, str)

    @pytest.mark.parametrize('name,model_type', model_by_name_data)
    def test_get_model_by_name(self, name, model_type):
        model, record, err = ModelApi.get_model_by_name(name)
        if model_type is not None:
            assert isinstance(record, Model) and err is None and model is not None
        else:
            assert isinstance(err, str) and model is None and record is None

    @pytest.mark.parametrize('model_type,model_class', model_type_data)
    def test_get_models_by_type(self, model_type, model_class):
        models, err = ModelApi.get_models_by_type(model_type)
        if not isinstance(model_type, str):
            assert models is None and isinstance(err, str)
        elif model_class is not None:
            assert isinstance(models, list) and err is None

            for model in models:
                assert model.type == model_type
        else:
            assert len(models) == 0

    @pytest.mark.parametrize('name,body,expected', train_model_data)
    def test_train_model(self, name, body, expected):
        result, err = ModelApi.train_model(name, body)

        assert result == expected

    def test_get_all_models(self):
        result = ModelApi.get_all_models()

        for x in result:
            assert isinstance(x, dict) and isinstance(x['name'], str) and isinstance(x['type'], str)
