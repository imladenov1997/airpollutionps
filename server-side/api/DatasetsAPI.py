import datetime
import json

import pandas

from airpyllution import DBManager
from airpyllution import MainTransformer
from airpyllution.DataTransformers.TransformerEnum import Transformers
from utils.Errors import Errors
from utils.Helpers import Helpers


class DatasetsApi:
    DATE_TIME_FORMAT = '%d-%m-%Y %H:%M'

    @staticmethod
    def get_dataset(body, use_dataframe=True):
        """
        Function for getting a dataset from database
        :param body: dict - requires several parameters:
          * type - of ML model (CNN, FullGP, etc.),
          * range - dict with start and end datetime strings in format Day-Month-Year H:M (24H format)
          * locations - list with lists of locations - list of location is a list with longitude and latitude, e.g.
          [longitude, latitude]
          * pollutant - name of the pollutant, e.g. PM10, PM2.5
          * data - dict with additional data such as weather data (data['weather'] is another dict)
        :param use_dataframe: bool - whether the returned dataset is a dataframe or a list
        :return: DataFrame | List | None
        """

        if not isinstance(body, dict):
            return None

        if 'range' not in body or 'locations' not in body or 'pollutant' not in body:
            return None

        if body['range'] is None or body['locations'] is None or body['pollutant'] is None:
            return None

        # if not isinstance('range', dict):
        #     return None
        #
        # if not isinstance(body['locations'], list):
        #     return None
        # else:
        #     result = list(filter(lambda c: not isinstance(c, list) or len(c) != 2, body['locations']))
        #     if len(result) != 0 and len(body['locations']) != 0:
        #         return None

        # Params required for the DBManager, acts as a config of a given dataset
        config_params = {
            "Date": DatasetsApi.DATE_TIME_FORMAT.split(' ')[0],
            "Time": DatasetsApi.DATE_TIME_FORMAT.split(' ')[1],
            "pollutant": {
                "Pollutant": None
            },
            'weather': {}
        }

        start_date = None
        end_date = None
        uncertainty = False

        if 'start' in body['range']:
            start_date = datetime.datetime.strptime(body['range']['start'], DatasetsApi.DATE_TIME_FORMAT)

        if 'end' in body['range']:
            end_date = datetime.datetime.strptime(body['range']['end'], DatasetsApi.DATE_TIME_FORMAT)

        if 'uncertainty' in body:
            uncertainty = True

        location_coordinates = []
        if isinstance(body['locations'], list):
            location_coordinates = list(map(lambda x: (x[0], x[1]), body['locations']))

        if isinstance(body['pollutant'], str):
            config_params['pollutant']['Pollutant'] = body['pollutant']

        if 'data' in body and isinstance(body['data'], dict):
            if 'weather' in body['data'] and isinstance(body['data']['weather'], dict):
                config_params['weather'] = body['data']['weather']

        datasets = []

        for coordinates_pair in location_coordinates:
            dataset, err = DBManager.get_dataset(datetime_from=start_date,
                                                 datetime_to=end_date,
                                                 longitude=coordinates_pair[0],
                                                 latitude=coordinates_pair[1],
                                                 config=config_params,
                                                 use_dataframe=use_dataframe,
                                                 uncertainty=uncertainty)

            dataset_size = len(dataset.index) if use_dataframe else len(dataset)

            if err is None and dataset_size != 0:
                datasets.append(dataset)

        if len(datasets) == 0:
            # TODO - IT IS VERY IMPORTANT TO CHANGE ALL CONDITIONS TO CHECK IF df.shape[0] == 0 IN THE API
            return pandas.DataFrame() if use_dataframe else []

        if use_dataframe:
            complete_dataset = pandas.concat(datasets)
            MainTransformer.periodic_f(complete_dataset)
        else:
            complete_dataset = []
            for x in datasets:
                complete_dataset.extend(x)

        return complete_dataset

    @staticmethod
    def insert_single_measurement(body):
        """
        Function for inserting a single measurement of pollution for a given date, time and location
        :param body: dict - requires several parameters:
          * type - of ML model (CNN, FullGP, etc.),
          * date_time - date and time of measurement
          * longitude - float
          * latitude - float
          * pollutant - name of the pollutant, e.g. PM10, PM2.5
          * pollution_value - float
          * data - dict with meteorological factors and and their values, e.g. data['Temperature'] = 3.3
        :param predicted: bool - whether the instance is a predicted or measured one
        :return: (True, None) | (False, str) - string instance is the error message
        """

        result, err = DatasetsApi.__are_params_valid(body)

        if not result:
            return result, err

        data = body['data'] if 'data' in body else None

        date_time = datetime.datetime.strptime(body['date_time'], DatasetsApi.DATE_TIME_FORMAT)

        is_successful, err = DBManager.insert_instance(longitude=body['longitude'], latitude=body['latitude'],
                                                       pollutant_name=body['pollutant'], predicted=False,
                                                       pollution_value=body['pollution_value'],
                                                       data=data, date_time=date_time)

        return is_successful, err

    @staticmethod
    def insert_single_prediction(body):
        """
        Function for inserting a single prediction of pollution for a given date, time and location
        :param body: dict - requires several parameters:
          * type - of ML model (CNN, FullGP, etc.),
          * date_time - date and time of measurement
          * longitude - float
          * latitude - float
          * pollutant - name of the pollutant, e.g. PM10, PM2.5
          * pollution_value - float
          * data - dict with meteorological factors and and their values, e.g. data['Temperature'] = 3.3
        :param predicted: bool - whether the instance is a predicted or measured one
        :return: (True, None) | (False, str) - string instance is the error message
        """

        result, err = DatasetsApi.__are_params_valid(body)

        if not result:
            return result, err

        data = body['data'] if 'data' in body else None

        date_time = datetime.datetime.strptime(body['date_time'], DatasetsApi.DATE_TIME_FORMAT)

        is_successful, err = DBManager.insert_prediction(longitude=body['longitude'], latitude=body['latitude'],
                                                         pollutant_name=body['pollutant'], predicted=True,
                                                         pollution_value=body['pollution_value'], date_time=date_time,
                                                         uncertainty=body['uncertainty'])

        return is_successful, err

    @staticmethod
    def insert_single_instance(body, predicted=False):
        """
        Function for inserting a single instance without a pollution value
        :param body: dict - requires several parameters:
          * date_time - date and time of measurement
          * longitude - float
          * latitude - float
          * pollutant - name of the pollutant, e.g. PM10, PM2.5
          * pollution_value - float
          * data - dict with meteorological factors and and their values, e.g. data['Temperature'] = 3.3
        :param predicted: bool - whether the instance is a predicted or measured one
        :return: (True, None) | (False, str) - string instance is the error message
        """

        result, err = DatasetsApi.__are_params_valid(body)

        if not result:
            return result, err

        data = body['data'] if 'data' in body else None
        pollutant_name = body['pollutant'] if 'pollutant' in body else None
        pollution_value = body['pollution_value'] if 'pollution_value' in body else None

        date_time = datetime.datetime.strptime(body['date_time'], DatasetsApi.DATE_TIME_FORMAT)

        if predicted is None:
            predicted = False

        is_successful, err = DBManager.insert_instance(longitude=body['longitude'], latitude=body['latitude'],
                                                       pollutant_name=pollutant_name, predicted=predicted,
                                                       pollution_value=pollution_value,
                                                       data=data, date_time=date_time)

        return is_successful, err

    @staticmethod
    def insert_dataset(files):
        """
        Function for inserting a whole dataset in the database
        :param files: dict with FileStorage instances, holding datasets' files
        :return: (True, None) | (False, str) - string instance is the error message
        """
        # parameters required for basic data such as which dataset to be improted, what time formats to be used, etc.
        BASE_PARAMS = [
            'Date',
            'Time'
        ]

        # parameters required for getting specific columns from given datasets, etc. for Temperature get tempC column
        DATASET_PARAMS = [
            'weatherFormat',
            'pollutantFormat'
        ]

        dataset_metadata = json.load(files['metadata'])

        if not isinstance(dataset_metadata, dict):
            return False, Errors.WRONG_INSTANCE.value

        are_params_missing = Helpers.are_params_missing(dataset_metadata, BASE_PARAMS + DATASET_PARAMS)

        if are_params_missing:
            return False, Errors.MISSING_PARAM.value

        for x in DATASET_PARAMS:
            if not isinstance(dataset_metadata[x], dict):
                return False, Errors.WRONG_INSTANCE.value

        for key in files:
            dataset_metadata[key + 'Datasets'] = files[key]

        # Combine multiple datasets and get result
        main_transformer = MainTransformer(config=dataset_metadata)
        main_transformer.add_transformer(Transformers.WEATHER_TRANSFORMER)
        main_transformer.add_transformer(Transformers.POLLUTANT_TRANSFORMER)
        main_transformer.transform()
        dataset = main_transformer.get_dataset()

        result, err = DBManager.insert_dataset(dataset, dataset_metadata)
        return result, err

    @staticmethod
    def get_coordinates():
        """
        Function for getting all coordinate pairs from DB
        :return: list of list of floats
        """
        coordinates = DBManager.get_all_coordinates()
        return coordinates

    @staticmethod
    def get_pollutants():
        """
        Function for getting all pollutants currently in the DB
        :return: list of str
        """
        pollutants = DBManager.get_pollutants()
        return pollutants

    @staticmethod
    def get_single_instance_dataset(body, stats=None, prev=None):
        """
        Function for generating a single instance dataset for CNN
        :param body: dict - parameters from request
        :param stats: dict - stats for dataset normalization on which model was trained
        :param prev: int - number of previous records to be generated
        :return: DataFrame with predictions
        """
        if not isinstance(body, dict):
            return None

        if 'date_time' not in body or not isinstance(body['date_time'], str):
            return None

        if 'longitude' not in body or not isinstance(body['longitude'], float):
            return None

        if 'latitude' not in body or not isinstance(body['latitude'], float):
            return None

        df_schema = {
            'DateTime': [body['date_time']],
            'Longitude': body['longitude'],
            'Latitude': body['latitude'],
            'Pollutant': None
        }

        instance_object = {
            'DateTime': body['date_time'],
            'Longitude': body['longitude'],
            'Latitude': body['latitude'],
            'Pollutant': None
        }

        data_keys = list()

        if 'data' in body and 'weather' in body['data']:
            for key in body['data']['weather'].keys():
                df_schema[key] = [body['data']['weather'][key]]
                instance_object[key] = body['data']['weather'][key]

        if isinstance(prev, int):
            ready_data = None
            DatasetsApi.generate_previous_records(df_schema, prev, ready_data)

        dataset = pandas.DataFrame(df_schema)
        automatic_normalization = not isinstance(stats, dict)  # if stats parameter is given
        dataset.set_index(keys='DateTime', inplace=True)

        MainTransformer.periodic_f(dataset)
        X_predict, _, _, _, _ = MainTransformer.get_training_and_test_set(dataset,
                                                                          'Pollutant',
                                                                          'Uncertainty',
                                                                          size=1,
                                                                          normalize=automatic_normalization)

        if not automatic_normalization:
            MainTransformer.normalize(X_predict, stats=stats, inplace=True)

        return X_predict

    @staticmethod
    def generate_previous_records(current_instance, n_prev, ready_data=None):
        """
        Function for actually generating previous records given a current instance
        ready_data should have dict as follows:
        [{date_time: ..., otherData...}]

        :param current_instance: dict
        :param n_prev: int - number of previous records
        :param ready_data: external data that is available for given previous number of records, such as meteorological
        factors
        :return:
        """
        if not isinstance(current_instance, dict):
            return None

        if 'DateTime' not in current_instance or not isinstance(current_instance['DateTime'][0], str):
            return None

        if 'Longitude' not in current_instance or not isinstance(current_instance['Longitude'], float):
            return None

        if 'Latitude' not in current_instance or not isinstance(current_instance['Latitude'], float):
            return None

        if 'Pollutant' not in current_instance:
            return None

        try:
            date_time = datetime.datetime.strptime(current_instance['DateTime'][0], DatasetsApi.DATE_TIME_FORMAT)
        except ValueError:
            return None

        date_times = []

        size = n_prev + 1

        for i in range(size):
            cur_date_time = date_time - datetime.timedelta(hours=i)
            date_times.insert(0, datetime.datetime.strftime(cur_date_time, DatasetsApi.DATE_TIME_FORMAT))

        current_instance['DateTime'] = date_times

        longitudes = [current_instance['Longitude'] for _ in range(size)]
        latitudes = [current_instance['Latitude'] for _ in range(size)]
        pollutant = [current_instance['Pollutant'] for _ in range(size)]

        current_instance['Longitude'] = longitudes
        current_instance['Latitude'] = latitudes
        current_instance['Pollutant'] = pollutant

        if isinstance(ready_data, list):
            for i in range(len(ready_data)):
                element = ready_data[len(ready_data) - i - 1]
                for key in element.keys():
                    if key in current_instance:
                        current_instance[key].insert(0)

        return current_instance

    @staticmethod
    def __are_params_valid(body):
        """
        Function to check if parameters are valid for processing a request
        :param body: dict - request body
        :return: (True, None) | (False, str) - str is the error message
        """
        if not isinstance(body, dict):
            return False, Errors.MISSING_BODY.value

        if 'date_time' not in body or not isinstance(body['date_time'], str):
            return False, Errors.MISSING_DATETIME.value

        if 'pollutant' not in body or not isinstance(body['pollutant'], str):
            return False, Errors.WRONG_LONGITUDE.value

        if 'pollution_value' not in body or not isinstance(body['pollution_value'], float):
            return False, Errors.WRONG_LONGITUDE.value

        return True, None
