import json

import pandas
import numpy as np

from .BaseSaver import AbstractSaver
from ..Utils.DateTimeUtils import DateTimeUtils

PREDICTOR_TYPE = 'JSON'


class JSONSaver(AbstractSaver):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.PREDICTOR_TYPE = PREDICTOR_TYPE

    def save_predictions(self, X_test, predictions):
        print('Saving predictions...')

        output = self.create_predictions_object(X_test, predictions)
        JSONSaver.__filesave(output, self.config['predictionFile'])
        print('Predictions saved...')

    def save_evaluations(self, X_test, predictions, target_values, metrics, error):
        print('Saving evaluations...')

        output = self.create_predictions_object(X_test, predictions, target=target_values)
        output[metrics] = error

        JSONSaver.__filesave(output, self.config['predictionFile'])
        print('Predictions saved...')

    def create_predictions_object(self, X_test, predictions, target=None):
        if self.config is None:
            print('No config set')
            return

        if not isinstance(X_test, pandas.DataFrame) or not isinstance(predictions, np.ndarray):
            return

        pollutant = self.config['pollutant']['Pollutant']
        output = {
            'pollutant': pollutant,
            'predictions': []
        }

        count = 0
        target_arr = None if target is None else target.to_numpy()
        for index, x in X_test.iterrows():
            if count >= len(predictions):
                break

            date, time = self.get_date_time(index)
            prediction = {
                'Time': time,
                'Date': date,
                'Location': {},
                'Prediction': str(predictions[count][0])
            }

            # Assume there is no location input in the dataset (when interpolating at one place)
            prediction['Location']['Longitude'] = None if 'Longitude' not in x else x['Longitude']
            prediction['Location']['Latitude'] = None if 'Latitude' not in x else x['Latitude']

            # Used only when outputting the error
            if target_arr is not None:
                prediction['Target'] = target_arr[count][0]

            output['predictions'].append(prediction)
            count += 1

        return output

    @staticmethod
    def __filesave(output, prediction_file):
        with open(prediction_file, 'w+') as json_obj:
            try:
                json.dump(output, json_obj, sort_keys=True, indent=4)
            except:
                print('Something went wrong with saving... Please, check permissions and filename in config.json')
                return

    def get_date_time(self, date_time):
        if not isinstance(date_time, str):
            return None, None

        [date, time] = date_time.split(' ')

        if DateTimeUtils.check_date_or_time(date, self.config['Date']):  # TODO Make the same for time
            return date, time

        return None, None
