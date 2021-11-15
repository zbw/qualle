#  Copyright 2021 ZBW – Leibniz Information Centre for Economics
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
import pytest
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.exceptions import NotFittedError

from qualle.label_calibration.category import MultiCategoryLabelCalibrator
from tests.common import DummyRegressor


@pytest.fixture
def calibrator():
    return MultiCategoryLabelCalibrator(DummyRegressor)


def test_mclc_fit_zero_categories_raises_value_error(calibrator, X):
    with pytest.raises(ValueError) as excinfo:
        calibrator.fit(X, np.array([]))
    assert str(excinfo.value) == 'Number of categories must be greater 0'


def test_mclc_predict(calibrator, X):
    calibrator.fit(X, np.array([[3, 5, 6], [1, 2, 7]]))
    assert (calibrator.predict(X) == [[0] * 3, [1] * 3]).all()


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


def test_mlc_fit_with_sparse_matrix_fits_calibrators_with_nparray(
        calibrator, X
):
    y = np.array([[3, 0], [0, 2]])
    y_coo = coo_matrix(y)

    calibrator.fit(X, y_coo)

    for i, clbtr in enumerate(calibrator.calibrators_):
        X_actual = clbtr.regressor.X
        y_actual = clbtr.regressor.y

        assert X_actual.shape[0] == len(X)
        assert type(y_actual) == np.ndarray
        assert (y_actual == y[:, i]).all()
