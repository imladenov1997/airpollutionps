import datetime
import json

import pandas
import pytest
import copy

import os
import sys

sys.path.append(os.getcwd())

from airpyllution.Utils.ConfigReader import ConfigReader
from airpyllution.Utils.ImageEncoder import ImageEncoder
from api import DatasetsApi

config = None
DBManager = None

config_reader_data = [
    ('non-existing', False),
    (123, False),
    (None, False),
    ('./airpyllution/config.json', True)
]

get_dataset_body = {
    "range": {
        "start": "01-01-2018 05:00",
        "end": "03-01-2018 05:00"
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

image_set_data = [
    (24, 25),
    (None, None),
    ('Test', None)
]


class TestUtils:
    @pytest.mark.parametrize('path,expected', config_reader_data)
    def test_config_reader(self, path, expected):
        result, err = ConfigReader.open_config(path)
        assert result == expected

    @pytest.mark.parametrize('seq_length,expected', image_set_data)
    def test_image_encoder_generate_image_set(self, seq_length, expected):
        dataset = DatasetsApi.get_dataset(get_dataset_body, use_dataframe=True)
        result = ImageEncoder.generate_image_set(dataset, seq_length)

        if result is not None:
            assert result.shape[0] == expected
        else:
            assert result is None




