import numbers
import pandas
from PIL import Image
import numpy as np


class ImageEncoder:
    """
    Class for transforming dataset into images
    """
    PIXEL_MAX = 255
    PIXEL_MIN = 0

    @staticmethod
    def generate_image_set(df, y, n_instances=None):
        """
        Function to generate image dataset
        :param df: DataFrame
        :param y: DataFrame
        :param n_instances: int - number of instances per image
        :return: np.array - with images
        """
        if not isinstance(y, int):
            return None

        img_arr = []
        size = df.shape[0]
        x = df.shape[1]
        for i in range(size - y):
            batch = df.iloc[i:i + y, :]
            img_arr.append(batch.to_numpy())

        return np.array(img_arr)

    @staticmethod
    def rescale_dataset(df, inplace=False):
        """
        Rescales a dataset in given range
        :param df: DataFrame
        :param inplace: bool - whether dataset should be changed inplace or a new one created
        :return:
        """
        processed = None
        if inplace:
            processed = df
        else:
            processed = pandas.DataFrame()
        for column in df:
            processed[column] = ImageEncoder.rescale_series(df[column])
        return processed

    @staticmethod
    def rescale_series(series):
        """
        Rescales a series of values, values are fixed
        :param series:
        :return:
        """
        max_val = series.max()
        min_val = series.min()

        old_range_lower = min_val + min_val * 0.3
        old_range_upper = max_val + max_val * 0.3
        ran = (old_range_lower, old_range_upper)
        return series.map(lambda x: ImageEncoder.rescale_func(ran, x))

    @staticmethod
    def rescale_func(old_range, val):
        """
        Actually does the rescaling
        :param old_range: float
        :param val: float
        :return: float - rescaled value
        """
        if not (isinstance(old_range[0], numbers.Number) and isinstance(old_range[1], numbers.Number)):
            return val
        """
        :param old_range - tuple (old_min, old_max)
        :param val - value to rescale
        }
        :return: rescaled val | original val
        """

        return int(ImageEncoder.PIXEL_MIN + (
        (ImageEncoder.PIXEL_MAX - ImageEncoder.PIXEL_MIN) * (val - old_range[0]) / (old_range[1] - old_range[0])))
