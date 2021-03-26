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
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError

from qualle.label_calibration.category import MultiCategoryLabelCalibrator
from tests.label_calibration.common import DummyRegressor


@pytest.fixture
def calibrator():
    return MultiCategoryLabelCalibrator(DummyRegressor(), 2)


def test_mclc_zero_categories_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        MultiCategoryLabelCalibrator(DummyRegressor(), 0)
    assert str(excinfo.value) == 'Number of categories must be greater 0'


def test_mclc_predict(calibrator, X):
    calibrator.fit(X, [[3, 5], [1, 2]])
    assert (calibrator.predict(X) == [[0, 1]] * 2).all()


def test_mclc_predict_without_fit_raises_exc(calibrator, X):
    with pytest.raises(NotFittedError):
        calibrator.predict(X)


def test_mclc_fit_fits_calibrators(calibrator, X, mocker):
    y = np.array([[3, 5], [1, 2]])
    spies = []
    for c in calibrator._calibrators:
        spies.append(mocker.spy(c, 'fit'))
    calibrator.fit(X, y)
    for i, spy in enumerate(spies):
        spy.assert_called_once()
        X_actual = spy.call_args[0][0]
        y_actual = spy.call_args[0][1]

        assert X_actual == X
        assert type(y_actual) == np.ndarray
        assert (y_actual == y[i]).all()
