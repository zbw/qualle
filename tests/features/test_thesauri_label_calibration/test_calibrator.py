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

from qualle.features.thesauri_label_calibration import ThesauriLabelCalibrator
import tests.features.test_thesauri_label_calibration.common as c


class DummyRegressor:

    X_total = []
    y_total = []

    def fit(self, X, y):
        DummyRegressor.X_total.append(X)
        DummyRegressor.y_total.append(y)

    def predict(self, X):
        return np.array(range(X.shape[0]))

    @classmethod
    def clear(cls):
        cls.X_total = []
        cls.y_total = []


@pytest.fixture
def calibrator(transformer):
    DummyRegressor.clear()
    transformer.fit()
    return ThesauriLabelCalibrator(transformer, DummyRegressor)


@pytest.fixture
def X():
    return ['doc0', 'doc1']


@pytest.fixture
def y():
    return [[c.CONCEPT_x0, c.CONCEPT_INVALID], [c.CONCEPT_x1, c.CONCEPT_x2]]


def test_fit_fits_underlying_regressors(calibrator, X, y):
    calibrator.fit(X, y)

    assert len(DummyRegressor.X_total) == 2
    assert len(DummyRegressor.y_total) == 2


def test_fit_fits_underlying_regressors_with_transformed_y(calibrator, X, y):
    calibrator.fit(X, y)

    assert len(DummyRegressor.X_total) == 2
    assert len(DummyRegressor.y_total) == 2
    assert (DummyRegressor.y_total[0] == np.array([1, 2])).all()
    assert (DummyRegressor.y_total[1] == np.array([0, 2])).all()


def test_fit_with_unfitted_transformer_raises_exc(transformer, X):
    with pytest.raises(NotFittedError):
        ThesauriLabelCalibrator(transformer).fit(X, [[c.CONCEPT_x0]] * 2)


def test_predict_without_fit_raises_exc(calibrator, X):
    with pytest.raises(NotFittedError):
        calibrator.predict(X)


def test_predict(calibrator, X, y):
    calibrator.fit(X, y)
    assert (calibrator.predict(X) == np.array([[0, 1]] * 2)).all()
