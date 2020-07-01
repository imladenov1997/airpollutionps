from datetime import datetime


class DateTimeUtils:
    def __init__(self):
        pass

    @staticmethod
    def check_date_or_time(date_or_time, format):
        """
        Function to check if date and time is in valid format
        :param date_or_time: str
        :param format: str
        :return: bool
        """
        try:
            datetime.strptime(date_or_time, format)
            return True
        except:
            return False
