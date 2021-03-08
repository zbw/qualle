#  Copyright (c) 2021 ZBW  – Leibniz Information Centre for Economics
#
#  This file is part of qualle.
#
#  qualle is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  qualle is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with qualle.  If not, see <http://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import pytest

from qualle.quality_estimation import RecallPredictor


class DummyEstimator:

    def fit(self, *args):
        # Empty because no functionality required
        pass

    def predict(self, X):
        return (
            X['label_calibration'].to_numpy() *
            X['no_of_pred_labels'].to_numpy()
        )


@pytest.fixture
def X():
    return pd.DataFrame(
        {
            'label_calibration': np.array([3, 1, 0], dtype="int32"),
            'no_of_pred_labels': np.array([2, 1, 5], dtype="int32")
        }
    )


@pytest.fixture
def predictor():
    return RecallPredictor(DummyEstimator())


def test_recall_predictor_predict(predictor, X):
    y = np.array([0.8, 1., 4])
    predictor.fit(X, y)

    assert np.array_equal(predictor.predict(X), np.array([3*2, 1*1, 0*5]))


def test_recall_predictor_fit(predictor, X, mocker):
    y = np.array([0.8, 1., 4])

    spy = mocker.spy(predictor.estimator, 'fit')
    predictor.fit(X, y)

    X_actual = spy.call_args[0][0]
    y_actual = spy.call_args[0][1]
    assert np.array_equal(X_actual, np.array([
        [3, 1], [1, 0], [0, -5]
    ]))
    assert np.array_equal(y_actual, y)
