import datetime
import json

import pandas
import pytest
import copy

import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from airpyllution import Metrics


rmse_data = [
    ([1, 2, 3], [3, 4, 5], 2.0),
    ([], [], np.math.nan),
    ('cool', [1, 2], np.math.nan),
    ([1, 2], 'cool', np.math.nan),
    ('cool', 'cool', np.math.nan),
    (1, 2, np.math.nan),
    (None, None, np.math.nan),
    (None, [1, 2], np.math.nan),
    ([1, 2], None, np.math.nan)
]

mape_data = [
    ([], [], np.math.nan),
    ('cool', [1, 2], np.math.nan),
    ([1, 2], 'cool', np.math.nan),
    ('cool', 'cool', np.math.nan),
    (1, 2, np.math.nan),
    (None, None, np.math.nan),
    (None, [1, 2], np.math.nan),
    ([1, 2], None, np.math.nan)
]

sse_data = [
    ([], [], np.math.nan),
    ('cool', [1, 2], np.math.nan),
    ([1, 2], 'cool', np.math.nan),
    ('cool', 'cool', np.math.nan),
    (1, 2, np.math.nan),
    (None, None, np.math.nan),
    (None, [1, 2], np.math.nan),
    ([1, 2], None, np.math.nan)
]


class TestMetrics:
    @pytest.mark.parametrize('list_one,list_two,expected', rmse_data)
    def test_rmse(self, list_one, list_two, expected):
        result = Metrics.rmse(list_one, list_two)
        if expected is np.math.nan:
            assert result is expected
        else:
            assert result == expected

    @pytest.mark.parametrize('list_one,list_two,expected', mape_data)
    def test_mape(self, list_one, list_two, expected):
        result = Metrics.mape(list_one, list_two)
        if expected is np.math.nan:
            assert result is expected
        else:
            assert result == expected

    @pytest.mark.parametrize('list_one,list_two,expected', sse_data)
    def test_sse(self, list_one, list_two, expected):
        result = Metrics.sse(list_one, list_two)
        if expected is np.math.nan:
            assert result is expected
        else:
            assert result == expected




