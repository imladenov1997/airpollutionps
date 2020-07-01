from functools import wraps

from sklearn.metrics import mean_squared_error
import numpy as np

import numbers


def check_int(f):
    """
    Wrapper function
    :param f: wrapped function
    :return: f: function
    """
    @wraps(f)
    def is_number(*args, **kwargs):
        for x in args:
            if isinstance(x, numbers.Number):
                return np.math.nan

        return f(*args, **kwargs)

    return is_number


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    @check_int
    def rmse(y_true, y_predicted):
        """
        Function for calculating Root Mean Squared Error (RMSE) when evaluating a model
        :param y_true: dataset with actual values
        :param y_predicted: dataset with predicted values
        :return: NaN | np.array
        """
        try:
            return np.sqrt(mean_squared_error(y_true, y_predicted))
        except ValueError:
            return np.math.nan
        except TypeError:
            return np.math.nan

    @staticmethod
    @check_int
    def mape(y_true, y_predicted):
        """
        Function for calculating Mean Absolute Percentage Error (MAPE) when evaluating a model
        :param y_true: dataset with actual values
        :param y_predicted: dataset with predicted values
        :return: NaN | np.array
        """
        try:
            return np.absolute((y_true - y_predicted) / y_true).mean() * 100
        except ValueError:
            return np.math.nan
        except TypeError:
            return np.math.nan

    @staticmethod
    @check_int
    def sse(y_true, y_predicted):
        try:
            return (y_true - y_predicted).sum()
        except ValueError:
            return np.math.nan
        except TypeError:
            return np.math.nan

    @staticmethod
    @check_int
    def mae(y_true, y_predicted):
        pass  # TODO implement MAE