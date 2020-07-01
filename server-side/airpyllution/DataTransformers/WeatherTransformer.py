from .BaseTransformer import BaseTransformer
import pandas

TRANSFORMER_TYPE = 'weather'


class WeatherTransformer(BaseTransformer):
    def __init__(self, dataset_path=None, fields_obj=None, date_format=None):
        super().__init__(dataset_path=dataset_path)
        self.raw_dataset = self.read_dataset(dataset_path)
        self.type = TRANSFORMER_TYPE
        self.weather_obj = fields_obj
        self.date_format = date_format

    def transform(self):
        """
        Process date and time in the input dataset as well as pick the features that were required in the config
        :return: DataFrame | list
        """
        if self.raw_dataset is None or self.weather_obj is None or self.date_format is None:
            return []

        raw_dataset_keys = set(self.raw_dataset.keys())

        if 'Time' not in self.weather_obj or 'Date' not in self.weather_obj:
            return []

        if self.weather_obj['Time'] not in raw_dataset_keys or self.weather_obj['Date'] not in raw_dataset_keys:
            return []

        self.processed_dataset = pandas.DataFrame()
        for factor in self.weather_obj.keys():
            self.processed_dataset[factor] = self.raw_dataset[self.weather_obj[factor]]

        keys = set(self.processed_dataset.keys())
        if 'Time' not in keys or 'Date' not in keys:
            return []

        self.processed_dataset['Time'] = self.processed_dataset['Time'].map(WeatherTransformer.convert_to_24h)

        if self.weather_obj['Date'] is not None and self.weather_obj['Time'] is not None:
            self.processed_dataset.insert(0, 'DateTime',
                                          self.processed_dataset['Date'] + ' ' + self.processed_dataset['Time'].astype(
                                              str))
            self.processed_dataset['DateTime'] = self.processed_dataset['DateTime'].map(self.uniform_date_time)
            self.processed_dataset['Time'] = self.processed_dataset['DateTime'].map(WeatherTransformer.get_time)
            pandas.to_datetime(self.processed_dataset['DateTime'])
            self.processed_dataset = self.processed_dataset.set_index('DateTime')
            self.processed_dataset = self.processed_dataset.drop(axis=1, columns=['Date', 'Time'])
        return self.processed_dataset