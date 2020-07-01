import datetime
import numpy as np
import pandas


class BaseTransformer:
    # static functions to get what's necessary from input
    FUNCS = {
        'day': (lambda x: BaseTransformer.split_date_time(x)[0]),
        'month': (lambda x: BaseTransformer.split_date_time(x)[1]),
        'year': (lambda x: BaseTransformer.split_date_time(x)[2]),
        'hour': (lambda x: BaseTransformer.split_date_time(x)[3]),
        'minute': (lambda x: BaseTransformer.split_date_time(x)[4]),
    }

    def __init__(self, dataset_path):
        self.raw_dataset = None  # input dataset
        self.config = None
        self.processed_dataset = []
        self.dataset_path = dataset_path
        self.date_format = {
            "Date": "%d-%m-%Y"
        }

    def show_dataset(self):
        """
        Print the dataset
        :return: None
        """
        if self.raw_dataset is not None:
            print(self.raw_dataset)

    def get_dataset(self):
        """
        Returns the dataset assigned to this transformer
        :return: DataFrame | None
        """
        return self.processed_dataset

    def empty_raw_dataset(self):
        """
        Remove raw dataset once it's transformed, no need to waste memory
        :return: None
        """
        self.raw_dataset = None

    def empty_processed_dataset(self):
        """
        Remove processed dataset
        :return: None
        """
        self.processed_dataset = None

    def empty_datasets(self):
        """
        Remove all datasets existing in this transformer
        :return: None
        """
        self.empty_processed_dataset()
        self.empty_raw_dataset()

    def transform(self):
        """
        Will be further extended
        :return:
        """
        pass

    def uniform_date_time(self, date_time):
        """
        Transform date and time into the standard format
        :param date_time: str
        :return: str
        """
        date_and_time = date_time.split(' ')
        date = date_and_time[0]
        time = date_and_time[1]
        date = datetime.datetime.strptime(date, self.date_format['Date'])

        if '24:' in date_time:
            date += datetime.timedelta(days=1)
            time = '00:00'

        date = date.strftime("%d-%m-%Y")

        return date + ' ' + time

    @staticmethod
    def get_time(date_time):
        """
        Get time from input string
        :param date_time: str
        :return: time: str - just time in standard format
        """
        date_and_time = date_time.split(' ')
        time = ''
        if len(date_and_time) == 2:
            time = date_and_time[1]

        return time

    @staticmethod
    def convert_to_24h(time):
        """
        Convert from format in HOURS to standard 24h format
        :param time: str
        :return: stringified_time: str - standard format 24h time
        """
        stringified_time = str(time)
        if ':' not in stringified_time:
            return HOURS[stringified_time]
        return stringified_time

    @staticmethod
    def getDayOfYear(date_time):
        """
        Get which consecutive day from the year given input is
        :param date_time: str
        :return: int - in the range between 1 and 365 - day of the year
        """
        date = date_time.split(' ')[0]
        converted_date_time = datetime.datetime.strptime(date, "%d-%m-%Y")
        return converted_date_time.timetuple().tm_yday

    @staticmethod
    def getHour(date_time):
        """
        Get hour from input date and time
        :param date_time: str
        :return: float: hours from the day
        """
        time = date_time.split(' ')[1].split(':')
        return float(time[0]) + float(time[1]) / 60

    @staticmethod
    def get_sine_hour(date_time):
        """
        Apply sine function to hour
        :param date_time: str
        :return: numpy.float in the range between -1 and 1 inclusive
        """
        return np.sin(np.pi * BaseTransformer.getHour(date_time) / 24) ** 2

    @staticmethod
    def get_sine_day_of_year(date_time):
        """
        Apply sine function to day of year
        :param date_time: str
        :return: numpy.float in the range between -1 and 1 inclusive
        """
        return np.sin(np.pi * BaseTransformer.getDayOfYear(date_time) / 365) ** 2

    @staticmethod
    def get_cosine_hour(date_time):
        """
        Apply cosine function to hour
        :param date_time: str
        :return: numpy.float in the range between -1 and 1 inclusive
        """
        return np.cos(np.pi * BaseTransformer.getHour(date_time) / 24) ** 2

    @staticmethod
    def get_cosine_day_of_year(date_time):
        """
        Apply cosine function to day of year
        :param date_time: str
        :return: numpy.float in the range between -1 and 1 inclusive
        """
        return np.cos(np.pi * BaseTransformer.getDayOfYear(date_time) / 365) ** 2

    @staticmethod
    def get_datetime_param(param):
        """
        Get some of the date or time parameters
        :param param: 'day', 'month', 'year', 'hour', 'minute': str, otherwise it throws error
        :return: str - given parameter from the above mentioned
        """
        return BaseTransformer.FUNCS[param]

    @staticmethod
    def split_date_time(raw_date_time):
        """
        Split date and time into separate components
        :param raw_date_time: str
        :return: list with split components in the following order: day, month, year, hour, minute
        """
        date_time = datetime.datetime.strptime(raw_date_time, "%d-%m-%Y %H:%M")
        return [
            date_time.day,
            date_time.month,
            date_time.year,
            date_time.hour,
            date_time.minute
        ]

    @staticmethod
    def read_dataset(path):
        """
        Read dataset from input path
        :param path: str
        :return: DataFrame | None
        """
        if isinstance(path, str):
            try:
                return pandas.read_csv(path)
            except:
                return None
        return None


HOURS = {
    '0': '00:00',
    '100': '01:00',
    '200': '02:00',
    '300': '03:00',
    '400': '04:00',
    '500': '05:00',
    '600': '06:00',
    '700': '07:00',
    '800': '08:00',
    '900': '09:00',
    '1000': '10:00',
    '1100': '11:00',
    '1200': '12:00',
    '1300': '13:00',
    '1400': '14:00',
    '1500': '15:00',
    '1600': '16:00',
    '1700': '17:00',
    '1800': '18:00',
    '1900': '19:00',
    '2000': '20:00',
    '2100': '21:00',
    '2200': '22:00',
    '2300': '23:00'
}
