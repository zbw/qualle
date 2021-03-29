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


class DummyRegressor:

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return np.array(range(X.shape[0]))


@pytest.fixture
def calibrator():
    return MultiCategoryLabelCalibrator(DummyRegressor)


def test_mclc_fit_zero_categories_raises_value_error(calibrator, X):
    with pytest.raises(ValueError) as excinfo:
        calibrator.fit(X, np.array([]))
    assert str(excinfo.value) == 'Number of categories must be greater 0'


def test_mclc_predict(calibrator, X):
    calibrator.fit(X, np.array([[3, 5, 6], [1, 2, 7]]))
    assert (calibrator.predict(X) == [[0, 1]] * 3).all()


def test_mclc_predict_without_fit_raises_exc(calibrator, X):
    with pytest.raises(NotFittedError):
        calibrator.predict(X)


def test_mclc_fit_fits_calibrators(calibrator, X):
    y = np.array([[3, 5], [1, 2]])

    calibrator.fit(X, y)

    for i, clbtr in enumerate(calibrator.calibrators_):
        X_actual = clbtr.regressor.X
        y_actual = clbtr.regressor.y

        assert X_actual.shape[0] == len(X)
        assert type(y_actual) == np.ndarray
        assert (y_actual == y[:, i]).all()
