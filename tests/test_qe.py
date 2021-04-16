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

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrationFeatures
from qualle.models import LabelCalibrationData
from qualle.quality_estimation import RecallPredictor


class DummyRegressor:

    def fit(self, *args):
        # Empty because no functionality required
        pass

    def predict(self, X):
        return X[:, 0] * X[:, 1]


@pytest.fixture
def X():
    return LabelCalibrationData(
        predicted_no_of_labels=np.array([3, 1, 1], dtype="int32"),
        predicted_labels=[['c'] * 2, ['c'], ['c'] * 5]
    )


@pytest.fixture
def predictor():
    return RecallPredictor(
        regressor=DummyRegressor(),
        features=SimpleLabelCalibrationFeatures()
    )


def test_rp_predict(predictor, X):
    y = np.array([0.8, 1., 4])
    predictor.fit(X, y)

    assert np.array_equal(
        predictor.predict(X),
        np.array([3 * (3 - 2), 1 * (1 - 1), 1 * (1 - 5)])
    )


def test_rp_fit_fits_regressor_with_features(predictor, X, mocker):
    y = np.array([0.8, 1., 4])
    X_transformed = predictor.features.transform(X)
    spy = mocker.spy(predictor.regressor, 'fit')
    predictor.fit(X, y)

    spy.assert_called_once()
    assert (spy.call_args[0][0] == X_transformed).all()
    assert (spy.call_args[0][1] == y).all()


def test_rp_predict_without_fit_raises_exc(predictor, X):
    with pytest.raises(NotFittedError):
        predictor.predict(X)
