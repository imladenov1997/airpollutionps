import os
import sys
import numpy as np
from datetime import datetime

import pandas
import pytest

sys.path.append(os.getcwd())

from airpyllution import WeatherTransformer
from airpyllution.DataTransformers.MainTransformer import MainTransformer, WrongConfigTypeException
from airpyllution.DataTransformers.TransformerEnum import Transformers
from airpyllution.Utils.ConfigReader import ConfigReader
from airpyllution import DB

ConfigReader.open_config()
config = ConfigReader.CONFIG


class TestFeatureTransformer:
    testing_data_weather = [
        (
            (config['weatherDatasets'], config['weather'], config['weatherFormat']),
            lambda x: isinstance(x, pandas.DataFrame)
        ),
        (
            (config['weatherDatasets'], None, config['weatherFormat']),
            lambda x: x == []
        ),
        (
            (config['weatherDatasets'], {}, config['weatherFormat']),
            lambda x: x == []
        ),
        (
            (None, config['weather'], config['weatherFormat']),
            lambda x: x == []
        ),
        (
            ({}, config['weather'], config['weatherFormat']),
            lambda x: x == []
        ),
        (
            (config['weatherDatasets'], config['pollutant'], None),
            lambda x: x == []
        ),
        (
            (config['weatherDatasets'], config['pollutant'], {}),
            lambda x: x == []
        ),
        (
            ('nice.csv', config['pollutant'], {}),
            lambda x: x == []
        )

    ]

    @pytest.mark.parametrize('data,func', testing_data_weather)
    def test_open_weather_dataset(self, data, func):
        pollutant_transformer = WeatherTransformer(
            dataset_path=data[0],
            fields_obj=data[1],
            date_format=data[2]
        )

        result = pollutant_transformer.transform()

        assert func(result)
