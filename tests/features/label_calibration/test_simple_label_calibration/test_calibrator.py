#  Copyright 2021-2025 ZBW – Leibniz Information Centre for Economics
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

from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrator,
)
from tests.common import DummyRegressor


@pytest.fixture
def calibrator():
    return SimpleLabelCalibrator(DummyRegressor())


@pytest.fixture
def X():
    return ["doc0", "doc1"]


@pytest.fixture
def y():
    return [["c0"], ["c0", "c1"]]


def test_fit_fits_underlying_regressor(calibrator, X, y):
    calibrator.fit(X, y)

    assert calibrator.regressor.X is not None
    assert calibrator.regressor.y is not None


def test_fit_fits_underlying_regressor_with_transformed_y(calibrator, X, y):
    calibrator.fit(X, y)

    assert calibrator.regressor.y is not None
    assert (calibrator.regressor.y == np.array([1, 2])).all()


def test_predict_without_fit_raises_exc(calibrator, X):
    with pytest.raises(NotFittedError):
        calibrator.predict(X)


def test_predict(calibrator, X, y):
    calibrator.fit(X, y)
    assert (calibrator.predict(X) == np.array([[0, 1]] * 2)).all()
