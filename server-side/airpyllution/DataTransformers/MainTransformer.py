import pandas

from .BaseTransformer import BaseTransformer
from .PollutantTransformer import PollutantTransformer
from .WeatherTransformer import WeatherTransformer
from .TransformerEnum import Transformers


class MainTransformer(BaseTransformer):
    DEFAULT_SIZE = 0.8  # size of dataset, value between 0-1

    def __init__(self, config=None):
        super().__init__(None)
        self.statistics = {}
        self.transformers = {}
        self.is_normalised = False
        self.__check_config(config) # raises WrongConfigTypeException and prevents instance from creation
        self.config = config

    def add_transformer(self, transformer_type):
        """
        Add a transformer in the object
        :param transformer_type: TransformerEnum
        :return: bool - whether adding was successful or not
        """
        if not isinstance(transformer_type, Transformers):
            return False

        if transformer_type.value not in self.transformers:
            arguments = {
                'dataset_path': self.config[(transformer_type.value + 'Datasets')],
                'fields_obj': self.config[transformer_type.value],
                'date_format': self.config[transformer_type.value + 'Format']
            }

            transformer = MainTransformer.init_transformer(transformer_type, arguments)

            if transformer is not None:
                self.transformers[transformer_type.value] = transformer
                return True

            return False

        return False

    def get_transformers(self):
        """
        Get transformers that are added to the MainTransformer
        :return: dict - all transformers
        """
        return self.transformers

    def init_transformers(self):
        """
        Add default transformers such as weather and pollution
        :return:
        """
        self.add_transformer(Transformers.WEATHER_TRANSFORMER)
        self.add_transformer(Transformers.POLLUTANT_TRANSFORMER)

    def transform(self):
        """
        Combine datasets from the existing transformers in self.transformers
        :return: DataFrame | None - with existing data
        """
        processed_sets = {}
        for key, transformer in self.transformers.items():
            processed_sets[key] = transformer.transform()

        self.processed_dataset = None

        for dataset in processed_sets.keys():
            if self.processed_dataset is not None:
                self.processed_dataset = self.processed_dataset.join(processed_sets[dataset])
            else:
                self.processed_dataset = processed_sets[dataset]

        self.periodic_f(self.processed_dataset)
        self.processed_dataset.dropna(inplace=True)

        return self.processed_dataset

    @staticmethod
    def periodic_f(dataset):
        """
        Add values of periodic functions applied to time and date
        :param dataset:
        :return: dataset - modified with 4 additional columns
        """
        dataset['TimeSin'] = dataset.index.map(MainTransformer.get_sine_hour)
        dataset['DateSin'] = dataset.index.map(MainTransformer.get_sine_day_of_year)
        dataset['TimeCos'] = dataset.index.map(MainTransformer.get_cosine_hour)
        dataset['DateCos'] = dataset.index.map(MainTransformer.get_cosine_day_of_year)

        return dataset

    @staticmethod
    def remove_periodic_f(dataset):
        dataset.drop(axis=1, columns=['TimeSin', 'TimeCos', 'DateSin', 'DateCos'], inplace=True)
        return dataset

    @staticmethod
    def normalize(dataset, stats=None, inplace=False):
        """"
        Normalises dataset given, could be used for both training and testing data (testing data must have statistics
        set)

        :param dataset - DataFrame that is the dataset itself, normalises all fields
        :param stats - None|dict (None because no prior knowledge before training set)
        :param inplace - boolean whether to edit the input dataset or copy it
        """
        if dataset is None:
            return None, None

        statistics = {} if stats is None else stats
        df = dataset if inplace else dataset.copy()

        for x in df.keys():
            if x == 'Pollutant':
                continue

            # Calculate mean and std if training set is used (as feature is not added in statistics)
            if x in statistics:
                mean = statistics[x]['mean']
                std = statistics[x]['std']
            else:
                mean = df[x].mean()
                std = df[x].std(ddof=0)

                statistics[x] = {
                    'mean': mean,
                    'std': std
                }

            df[x] -= mean
            df[x] /= std if std != 0 else 1

        return df, statistics

    @staticmethod
    def unnormalize(dataset, statistics, inplace=False):
        """
        Restore dataset to its original form so that it can be further saved
        :param dataset: DataFrame
        :param statistics: dict
        :param inplace: bool
        :return: DataFrame | None
        """
        if dataset is None:
            return None

        if statistics is None:
            return None

        df = dataset if inplace else dataset.copy()

        for x in dataset.keys():
            if x == 'Pollutant':
                continue

            if x in statistics:
                df[x] *= statistics[x]['std']
                df[x] += statistics[x]['mean']

        return df

    def insert_location(self):
        """
        Insert location to the final dataset if it exists in the existing datasets
        :return: DataFrame
        """
        if 'Location' not in self.config:
            return self.processed_dataset
        if 'Longitude' not in self.config['Location'] or 'Latitude' not in self.config['Location']:
            return self.processed_dataset

        longitude = self.config['Location']['Longitude']
        latitude = self.config['Location']['Latitude']

        if longitude < 0:
            longitude += 360

        if latitude < 0:
            latitude += 360

        if self.processed_dataset is not None:
            if 'Longitude' not in self.processed_dataset.columns and 'Latitude' not in self.processed_dataset.columns:
                self.processed_dataset.insert(3, 'Longitude', longitude)
                self.processed_dataset.insert(4, 'Latitude', latitude)
        return self.processed_dataset

    def get_training_test_set(self, size=None, normalize=True):
        pollutant_name = 'Pollutant'  # according to standard, pollution levels' feature will have this name
        uncertainty_name = 'Uncertainty'  # according to standard, uncertainty feature will have this name
        return self.get_training_and_test_set(self.processed_dataset, pollutant_name, uncertainty_name,
                                              size=size, normalize=normalize)

    @staticmethod
    def get_training_and_test_set(dataset, pollutant_name_feature, uncertainty_name, size=None, normalize=True):
        """
        Used for splitting the dataset into training and test, making it possible to feed it into models
        :param dataset: DataFrame
        :param pollutant_name_feature: str - name of pollutant feature in the dataset
        :param uncertainty_name: str - name of uncertainty feature in the dataset
        :param size: int - size of training set
        :param normalize: boolean - whether to normalize dataset
        :return: X_train, y_train, X_test, y_test, stats - stats is the statistics necessary for normalizing further
        datasets of unnormalizing current dataset
        """
        n_instances = int(size * dataset.shape[0]) if size is not None else MainTransformer.DEFAULT_SIZE * \
                                                                            dataset.shape[0]

        # Get training set
        X_train = dataset.drop(axis=1, columns=[pollutant_name_feature], inplace=False).iloc[:n_instances]
        if uncertainty_name in X_train:
            X_train.drop(axis=1, columns=[uncertainty_name], inplace=True)

        y_train = dataset[[pollutant_name_feature]].iloc[:n_instances]

        stats = None
        if normalize:
            _, stats = MainTransformer.normalize(X_train, stats=None, inplace=True)

        # Get test set
        X_test = dataset.drop(axis=1, columns=[pollutant_name_feature], inplace=False).iloc[n_instances:]
        X_test = X_test.drop(axis=1, columns=[uncertainty_name], inplace=True) if uncertainty_name in X_test else X_test
        y_test = dataset[[pollutant_name_feature]].iloc[n_instances:]

        if normalize and stats is not None:
            MainTransformer.normalize(X_test, stats=stats, inplace=True)

        return X_train, y_train, X_test, y_test, stats

    @staticmethod
    def init_transformer(transformer_type, kwargs):
        """
        Initialise new transformer using factory pattern
        :param transformer_type:
        :param kwargs: dict
        :return: BaseTransformer | None
        """
        if not isinstance(transformer_type, Transformers):
            return None

        if not isinstance(kwargs, dict):
            return None

        if transformer_type.WEATHER_TRANSFORMER: return WeatherTransformer(**kwargs)
        if transformer_type.POLLUTANT_TRANSFORMER: return PollutantTransformer(**kwargs)

        return None

    @staticmethod
    def remove_features(dataset, elements_to_remove):
        """
        Remove selected features from input dataset, changes the dataset inplace
        :param dataset: DataFrame
        :param elements_to_remove: list
        :return: DataFrame - new dataset
        """
        dataset.drop(axis=1, columns=elements_to_remove, errors='ignore', inplace=True)
        return dataset

    @staticmethod
    def normalize_with_old_stats(n_cur_instances, old_stats, dataset):
        n_new_instances = dataset.shape[0]
        total = n_cur_instances + n_new_instances

        # Get weighted average of both
        weight_current_instances = n_cur_instances / total
        weight_new_instances = n_new_instances / total

        updated_dataset_stats = {}

        _, new_stats = MainTransformer.normalize(dataset, stats=None, inplace=False)

        for key, value in old_stats.items():
            if key in new_stats and 'mean' in new_stats[key] and 'std' in new_stats[key]:
                updated_dataset_stats[key] = {}
                updated_dataset_stats[key]['mean'] = new_stats[key]['mean'] * weight_new_instances
                updated_dataset_stats[key]['std'] = new_stats[key]['std'] * weight_new_instances
            else:
                continue

            if 'mean' in value and 'std' in value:
                updated_dataset_stats[key]['mean'] += value['mean'] * weight_current_instances
                updated_dataset_stats[key]['std'] += value['std'] * weight_current_instances

        return updated_dataset_stats, new_stats

    @staticmethod
    def __check_config(config):
        """
        Check if config is loaded
        :param config: dict
        :return: None
        """
        if not isinstance(config, dict):
            raise WrongConfigTypeException()


class WrongConfigTypeException(Exception):
    """
    Class for raising exception for wrong type of config
    """
    def __init__(self):
        super().__init__('Missing or Wrong Config, Config must be a dict')


class MissingConfigParamException(Exception):
    """
    Class for raising exception when there is a missing config
    """
    def __init__(self, message):
        super().__init__('Missing param in MainTransformer config: ' + message)
