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

from qualle.features.simple_label_calibration import SimpleLabelCalibrator
from tests.common import DummyRegressor


@pytest.fixture
def calibrator():
    return SimpleLabelCalibrator(DummyRegressor())


@pytest.fixture
def X():
    return ['doc0', 'doc1']


@pytest.fixture
def y():
    return [['c0'], ['c0', 'c1']]


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
