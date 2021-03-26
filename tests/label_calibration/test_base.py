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

from qualle.label_calibration.base import LabelCalibrator
from tests.label_calibration.common import DummyRegressor


@pytest.fixture
def calibrator():
    return LabelCalibrator(DummyRegressor())


def test_lc_predict(calibrator, X):
    calibrator.fit(X, [3, 5])
    assert np.array_equal(calibrator.predict(X), [0, 1])


def test_lc_predict_without_fit_raises_exc(calibrator, X):
    with pytest.raises(NotFittedError):
        calibrator.predict(X)


def test_lc_fit_fits_pipeline(calibrator, X, mocker):
    y = [3, 5]
    spy = mocker.spy(calibrator._pipeline, 'fit')
    calibrator.fit(X, y)
    spy.assert_called_with(X, y)
