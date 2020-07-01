import json
import pandas
import os
from datetime import datetime
from playhouse.postgres_ext import PostgresqlExtDatabase, Cast, JOIN, IntegrityError
from ..Utils.Errors import Errors
from .tables.base import base
from .tables.dataset import dataset as dataset_model
from .tables.pollution_level import pollution_level
from .tables.pollutant import pollutant
from .tables.mlmodel import ml_model


class Tables:
    """
    Class Tables contains all classes for each table
    Used for easier referencing and extendability
    Each variable in this class MUST reference a class mapped to a table in the database
    """
    Dataset = None
    PollutionLevel = None
    Pollutant = None
    MLModel = None


class DBManager:
    """
    Main class that deals with all queries to the database
    """
    DATE_TIME_FORMAT = '%Y-%m-%d %H:%M'
    __DB = None
    __tables = {
        'datasets': None,
        'pollutants': None,
        'pollution_level': None,
        'ml_models': None
    }

    @staticmethod
    def create_tables():
        """
        Function that creates all the tables if they don't exist
        :return:
        """
        if DBManager.__DB is not None:
            Base = base(DBManager.__DB)
            Tables.Dataset = Dataset = dataset_model(Base)
            Tables.Pollutant = Pollutant = pollutant(Base)
            Tables.PollutionLevel = PollutionLevel = pollution_level(Base, Dataset=Dataset, Pollutant=Pollutant)
            Tables.MLModel = MLModel = ml_model(Base)

            DBManager.__DB.create_tables([Dataset, Pollutant, PollutionLevel, MLModel])

    @staticmethod
    def connect():
        """
        Function that connects to the database given the credentials in DB/db.json
        Credentials are not kept in memory this way
        :return:
        """
        try:
            DBManager.__DB = PostgresqlExtDatabase(database=os.environ['DB'], user=os.environ['USERNAME'],
                                                   password=os.environ['PASSWORD'], host=os.environ['HOST'],
                                                   port=os.environ['PORT'])
            print('Connected.')
            return True, None
        except:
            return False, Errors.DB_CONN_FAIL.value

    @staticmethod
    def insert_dataset(dataset, config, predicted=False):
        """"
            Insert a whole dataset
            :param dataset - DataFrame with dataset to be inserted
            :param config - dict object with necessary fields to be conidered as config, pollutant, Date and Time fields
            are required
            :param predicted boolean - whether the pollution values in the dataset are predicted or measured
            :return (True, None) if successful, (False, String) if unsuccessful, string value is the error message
        """
        data_keys = list()
        pollutant_name = None

        # Weather data
        if 'weather' in config:
            data_keys += [x for x in config['weather'] if x != 'Time' and x != 'Date']

        # pollutant's feature name in the dataset
        if 'pollutant' in config:
            pollutant_name = config['pollutant']['Pollutant']

        with DBManager.__DB.atomic() as transaction:
            # First get or insert the pollutant and return its PK

            if isinstance(pollutant_name, str):
                pk_pollutant, _ = Tables.Pollutant.get_or_create(name=pollutant_name)
            else:
                transaction.rollback()
                return False, Errors.MISSING_PARAM.value

            for index, row in dataset.iterrows():
                data = dict()
                for x in data_keys:
                    data[x] = row[x]

                date_time = index if isinstance(index, str) else row['DateTime']

                # Insert instance
                pk_instance = Tables.Dataset.create(
                    datetime=datetime.strptime(date_time, config['Date'] + ' ' + config['Time']).strftime(
                        DBManager.DATE_TIME_FORMAT),
                    longitude=row['Longitude'],
                    latitude=row['Latitude'],
                    data=data
                )

                # Finally insert value of instance
                Tables.PollutionLevel.create(
                    dataset_id=pk_instance,
                    pollutant_id=pk_pollutant,
                    pollutant_value=row['Pollutant'],
                    predicted=predicted
                )

        return True, None

    @staticmethod
    def get_dataset(datetime_from=None, datetime_to=None, longitude=None, latitude=None, config=None,
                    use_dataframe=True, uncertainty=False):
        """"
        Function for getting dataset in given time period and location (if given)
        :param datetime_from - datetime object with both date and time
        :param datetime_to - datetime object with both date and time
        :param longitude - float
        :param latitude - float
        :param config - dict - it requires pollutant, Date, Time fields
        :return (DataFrame|list, None) | (None, str) - str is the error message
        """
        if not (datetime_to is None or isinstance(datetime_to, datetime)):
            return None, Errors.NO_DATETIME.value

        if not (datetime_from is None or isinstance(datetime_from, datetime)):
            return None, Errors.NO_DATETIME.value

        if not (
                    (longitude is None or isinstance(longitude, float)) or
                    (latitude is None or isinstance(latitude, float))
        ):
            return None, Errors.NO_LOCATION.value

        if config is None or not isinstance(config, dict):
            return None, Errors.WRONG_CONFIG.value

        if 'Date' not in config or 'Time' not in config:
            return None, Errors.MISSING_CONFIG_DATA.value

        pollutant_join_conds = None
        pollution_level_join_conds = None
        location_conds = True

        if 'pollutant' in config:
            pollutant_name = config['pollutant']['Pollutant']
            pollutant_join_conds = \
                Tables.Pollutant.id == Tables.PollutionLevel.pollutant_id

        if datetime_from is not None and datetime_to is not None:
            datetime_expression = Tables.Dataset.datetime.between(datetime_from, datetime_to)
        elif datetime_to is not None:
            datetime_expression = Cast(datetime_to, 'timestamp') >= Tables.Dataset.datetime
        elif datetime_from is not None:
            datetime_expression = Cast(datetime_from, 'timestamp') <= Tables.Dataset.datetime
        else:
            return [], Errors.MISSING_PARAM.value

        if isinstance(longitude, float) and isinstance(latitude, float):
            location_conds = (Tables.Dataset.longitude == Cast(longitude, 'double precision')) & \
                             (Tables.Dataset.latitude == Cast(latitude, 'double precision'))
        elif isinstance(longitude, float):
            location_conds = (Tables.Dataset.longitude == Cast(longitude, 'double precision'))
        elif isinstance(latitude, float):
            location_conds = (Tables.Dataset.latitude == Cast(latitude, 'double precision'))

        args = [
            Tables.Dataset,
            Tables.PollutionLevel.pollutant_value,
            Tables.Pollutant.name.alias('pollutant_name')
        ]

        if uncertainty:
            args.append(Tables.PollutionLevel.uncertainty)

        results = Tables.Dataset.select(*args) \
            .join(Tables.PollutionLevel, JOIN.LEFT_OUTER, on=pollution_level_join_conds) \
            .join(Tables.Pollutant, JOIN.LEFT_OUTER, on=pollutant_join_conds) \
            .where(datetime_expression & location_conds) \
            .order_by(Tables.Dataset.datetime.asc())

        # results = DBManager.__tables['pollution_level'].select()
        data_keys = list()

        if 'weather' in config:
            data_keys += [x for x in config['weather'] if x != 'Time' and x != 'Date']

        df_schema = {
            'DateTime': [],
            'Longitude': [],
            'Latitude': [],
            'Pollutant': []
        }

        for key in data_keys:
            df_schema[key] = []

        if uncertainty:
            df_schema['Uncertainty'] = []

        if use_dataframe:
            df = pandas.DataFrame(df_schema)
            count = 0

            for row in results.namedtuples():
                instance = {
                    'DateTime': row.datetime.strftime(config['Date'] + ' ' + config['Time']),
                    'Longitude': row.longitude,
                    'Latitude': row.latitude,
                    'Pollutant': row.pollutant_value
                }

                for key in data_keys:
                    instance[key] = row.data[key] if key in row.data else None

                if uncertainty:
                    instance['Uncertainty'] = row.uncertainty if row.uncertainty is not None else 0

                df.loc[count] = instance
                count += 1
            df.reset_index(drop=True, inplace=True)
            # Sort by DateTime
            df.set_index('DateTime', inplace=True)
            return df, None
        else:
            dataset_list = []
            for row in results.namedtuples():
                instance = {
                    'DateTime': row.datetime.strftime(config['Date'] + ' ' + config['Time']),
                    'Longitude': row.longitude,
                    'Latitude': row.latitude,
                    'Pollutant': row.pollutant_value
                }

                for key in data_keys:
                    instance[key] = row.data[key] if key in row.data else None

                if uncertainty:
                    instance['Uncertainty'] = row.uncertainty if row.uncertainty is not None else 0

                dataset_list.append(instance)

            return dataset_list, None

    @staticmethod
    def insert_instance(date_time=None, longitude=None, latitude=None, pollution_value=None,
                        data=None, predicted=False, pollutant_name=None):
        """"
            Function for inserting a single instance (measurement) for a given time and location
            :param date_time - datetime object
            :param longitude - float
            :param latitude - float
            :param pollution_value - float | None - level of pollution for given 4D space, it may not be existing,
            measurement so it can be left as None
            :param data - external metadata such as weather, etc.
            :param predicted - boolean - whether the pollution_value is a predicted one or measured one
            :param pollutant_name - str - what type of pollution
            :return (True, None) or (False, str) where str is the error message
        """
        if date_time is None or not isinstance(date_time, datetime):
            return False, Errors.NO_DATETIME.value

        if longitude is None or latitude is None or not (isinstance(longitude, float) or isinstance(latitude, float)):
            return False, Errors.NO_LOCATION.value

        if data is None or not isinstance(data, dict):
            data = {}

        with DBManager.__DB.atomic() as transaction:
            # Insert instance
            pk_instance = Tables.Dataset.create(
                datetime=datetime.strftime(date_time, DBManager.DATE_TIME_FORMAT),
                longitude=longitude,
                latitude=latitude,
                data=data
            )

            if isinstance(pollutant_name, str):
                pk_pollutant, _ = Tables.Pollutant.get_or_create(name=pollutant_name)
            else:
                # if no type of pollution, not sure what pollutant we are inserting
                transaction.commit()
                return True, Errors.NO_POLLUTANT.value

            # We may have inserted the type of pollution to have it, but we need also its value
            if isinstance(pollution_value, float) and isinstance(predicted, bool):
                Tables.PollutionLevel.create(
                    dataset_id=pk_instance,
                    pollutant_id=pk_pollutant,
                    pollutant_value=pollution_value,
                    predicted=predicted
                )
            else:
                return True, Errors.WRONG_INSTANCE.value

        return True, None

    @staticmethod
    def upsert_model(name, model_type, resource, model_params=None, extra_data=None, extra_params=None):
        """"
        Function for inserting a (trained) model to the database
        :param name - str - unique name of the model, arbitrarily given, should be *unique*
        :param model_type - str - type of model (e.g. CNN, GP, SparseGP, etc.)
        :param resource - str - resource library/framework which handles the generic model
        :param model_params - dict - a set of model parameters such as weights, architecture (for NNs), etc.
        :param extra_data - File - extra data for the model which is stored in a file (e.g. h5)
        :param extra_params - extra metadata for building the model (such as number of features in the dataset it has
        been trained, etc.)
        :return (True, None) | (False, str) - string is the error message
        """
        if not (isinstance(name, str) and isinstance(model_type, str) and isinstance(resource, str)):
            return False, Errors.WRONG_INSTANCE.value

        if model_params is not None and not isinstance(model_params, str):
            return False, Errors.WRONG_INSTANCE.value

        if extra_params is not None and not isinstance(extra_params, str):
            return False, Errors.WRONG_INSTANCE.value

        if extra_params is None:
            extra_params = json.dumps({})

        if extra_data is None:
            extra_data = json.dumps({})

        if model_params is None:
            model_params = json.dumps({})


        print(extra_params)

        with DBManager.__DB.atomic():
            Tables.MLModel.insert(
                name=name,
                type=model_type,
                resource=resource,
                extra_params=extra_params,
                model_params=model_params,
                extra_data=extra_data
            ).on_conflict(conflict_target=Tables.MLModel.name, update={
                Tables.MLModel.extra_params: extra_params,
                Tables.MLModel.model_params: model_params,
                Tables.MLModel.extra_data: extra_data
            }).execute()

        return True, None

    @staticmethod
    def get_model_by_name(name):
        """
        Get a model by its name
        :param name: str - name of the required model
        :return: (MLModel, None) | (None, str) where str value is an error message
        """
        if isinstance(name, str):
            name_expr = Tables.MLModel.name == name
            try:
                model_record = Tables.MLModel.get(name_expr)
            except:
                return None, Errors.NO_SUCH_MODEL.value
            return model_record, None

        return None, Errors.WRONG_INSTANCE.value

    @staticmethod
    def get_all_models():
        result = Tables.MLModel.select(Tables.MLModel.name, Tables.MLModel.type)

        models = []
        for model in result.namedtuples():
            model_dict = {
                'name': model.name,
                'type': model.type
            }

            models.append(model_dict)

        return models

    @staticmethod
    def get_models_metadata_by_type(model_type):
        """
        Function for getting a model by its type (e.g. Gaussian Processes, CNN, etc)
        :param model_type: str - type of the model
        :return: list | None - for schema data - DB.tables.mlmodel or check ERD
        """

        if isinstance(model_type, str):
            type_expr = Tables.MLModel.type == model_type
            model_records = Tables.MLModel.select(
                Tables.MLModel.name,
                Tables.MLModel.type,
                Tables.MLModel.resource
            ).where(type_expr)
            models_metadata = []
            for x in model_records.namedtuples():
                models_metadata.append(x)

            return models_metadata, None

        return None, Errors.WRONG_INSTANCE.value

    @staticmethod
    def get_models_metadata_by_resource(resource):
        """"
        Function for getting all metadata for models that are using given library (e.g. keras, GPy, scikit-learn...)

        The reason for getting only the metadata is the potential memory that the returned result may take
        Some models can take up a lot of memory, and a bunch of them would cause taking too much RAM

        Better to get only a their metadata and then select by name or type

        :param resource: str - name of the resource
        :return MLModel | None - trained model, its data can be seen in DB.tables.mlmodel or check ERD
        """

        if isinstance(resource, str):
            resource_expr = Tables.MLModel.resource == resource
            model_records = Tables.MLModel.select(
                Tables.MLModel.name,
                Tables.MLModel.type,
                Tables.MLModel.resource
            ).where(resource_expr)
            models_metadata = []
            for x in model_records.namedtuples():
                models_metadata.append(x)

            return models_metadata, None

        return None, Errors.WRONG_INSTANCE.value

    @staticmethod
    def insert_prediction(date_time=None, longitude=None, latitude=None, pollution_value=None, predicted=True,
                          pollutant_name=None, uncertainty=None):
        """
        Function for inserting a single prediction value to an exiting instance of the dataset,
        the instance must exist in the database, otherwise use insert_instance()
        :param uncertainty:
        :param date_time: datetime object ahving the date and time of the measurement
        :param longitude: float
        :param latitude: float
        :param pollution_value: float
        :param predicted: boolean - whether the value was predicted or added as a result of measurement
        :param pollutant_name: str - name of the pollution
        :return: (False, str) | (True, None) - str is the error message
        """
        if date_time is None or not isinstance(date_time, datetime):
            return False, Errors.NO_DATETIME.value

        if longitude is None or latitude is None or not (isinstance(longitude, float) or isinstance(latitude, float)):
            return False, Errors.NO_LOCATION.value

        if not isinstance(pollutant_name, str):
            return False, Errors.WRONG_INSTANCE.value

        if not isinstance(pollution_value, float):
            return False, Errors.WRONG_INSTANCE.value

        if not isinstance(predicted, bool):
            return False, Errors.PREDICTED_NOT_BOOL.value

        with DBManager.__DB.atomic() as transaction:
            # Get pollutant from the database or create if it does not exist,
            # its pk is used as a foreign key in pollution_level
            try:
                pk_pollutant, _ = Tables.Pollutant.get_or_create(name=pollutant_name)
            except:
                transaction.rollback()
                return False, Errors.POLLUTANT_INSERTION_OR_RETRIEVAL_FAILED.value

            # Find the data instance
            data_instance = Tables.Dataset.select().where(
                (Tables.Dataset.datetime == datetime.strftime(date_time, DBManager.DATE_TIME_FORMAT)) &
                (Tables.Dataset.longitude == Cast(longitude, 'double precision')) &
                (Tables.Dataset.latitude == Cast(latitude, 'double precision'))
            )

            print(uncertainty)

            for row in data_instance.namedtuples():
                # workaround for missing on_conflict on composite keys, query builder in use has nothin on constraints
                # for composite keys, not really preferable workaround, but still works
                try:
                    Tables.PollutionLevel.get_or_create(
                        dataset_id=row.id,
                        pollutant_id=pk_pollutant,
                        pollutant_value=pollution_value,
                        predicted=predicted,
                        uncertainty=uncertainty
                    )
                except IntegrityError:
                    Tables.PollutionLevel.update(
                        pollutant_value=pollution_value,
                        predicted=predicted,
                        uncertainty=uncertainty
                    ).where(
                        (Tables.PollutionLevel.dataset_id == row.id) &
                        (Tables.PollutionLevel.pollutant_id == pk_pollutant)
                    ).execute()
                except:
                    transaction.rollback()
                    return False, Errors.POLLUTION_LEVEL_INSERTION_FAILED.value

            return True, None

    @staticmethod
    def get_all_coordinates():
        """
        Get all coordinate pairs existing in the database
        :return: list of list, inner list containing 2 values - longitude and latitude
        """
        coordinates = Tables.Dataset.select(Tables.Dataset.longitude, Tables.Dataset.latitude).distinct()
        coordinates_list = []
        for pair in coordinates.namedtuples():
            coordinates_list.append([pair.longitude, pair.latitude])

        return coordinates_list

    @staticmethod
    def get_pollutants():
        """
        Get list of pollutants in the database
        :return: list of str
        """
        pollutants = Tables.Pollutant.select(Tables.Pollutant.name)
        pollutants_list = []
        for pollutant_instance in pollutants.namedtuples():
            pollutants_list.append(pollutant_instance.name)

        return pollutants_list

    @staticmethod
    def set_db(db):
        pass
