#  Copyright 2021-2023 ZBW – Leibniz Information Centre for Economics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qualle.features.label_calibration.simple import (
    SimpleLabelCalibrationFeatures,
)
from qualle.models import LabelCalibrationData
from qualle.quality_estimation import QualityEstimator


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
        predicted_labels=[["c"] * 2, ["c"], ["c"] * 5],
    )


@pytest.fixture
def predictor():
    return QualityEstimator(
        regressor=DummyRegressor(), features=SimpleLabelCalibrationFeatures()
    )


def test_rp_predict(predictor, X):
    y = np.array([0.8, 1.0, 4])
    predictor.fit(X, y)

    assert np.array_equal(
        predictor.predict(X), np.array([3 * (3 - 2), 1 * (1 - 1), 1 * (1 - 5)])
    )


def test_rp_fit_fits_regressor_with_features(predictor, X, mocker):
    y = np.array([0.8, 1.0, 4])
    X_transformed = predictor.features.transform(X)
    spy = mocker.spy(predictor.regressor, "fit")
    predictor.fit(X, y)

    spy.assert_called_once()
    assert (spy.call_args[0][0] == X_transformed).all()
    assert (spy.call_args[0][1] == y).all()


def test_rp_predict_without_fit_raises_exc(predictor, X):
    with pytest.raises(NotFittedError):
        predictor.predict(X)
