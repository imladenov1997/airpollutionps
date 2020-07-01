from .BaseTransformer import BaseTransformer
import pandas

TRANSFORMER_TYPE = 'pollutant'


class PollutantTransformer(BaseTransformer):
    def __init__(self, dataset_path=None, fields_obj=None, date_format=None):
        super().__init__(dataset_path)
        self.type = TRANSFORMER_TYPE
        self.pollutant_obj = fields_obj
        self.date_format = date_format

    def transform(self):
        """
        Process date and time in the input dataset
        :return: DataFrame
        """
        if self.pollutant_obj['Date'] is not None and self.pollutant_obj['Time'] is not None:
            self.processed_dataset.insert(0, 'DateTime',
                                          self.processed_dataset['Date'] + ' ' + self.processed_dataset['Time'].astype(
                                              str))
            self.processed_dataset['DateTime'] = self.processed_dataset['DateTime'].map(self.uniform_date_time)
            self.processed_dataset['Time'] = self.processed_dataset['DateTime'].map(PollutantTransformer.get_time)
            pandas.to_datetime(self.processed_dataset['DateTime'])
            self.processed_dataset.set_index('DateTime')
            self.processed_dataset = self.processed_dataset.drop(axis=1, columns=['Date', 'Time'])
        return self.processed_dataset

    def get_pollutant_dataset(self):
        return self.processed_dataset
