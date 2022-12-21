#  Copyright 2021-2022 ZBW – Leibniz Information Centre for Economics
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

from qualle.features.label_calibration.thesauri_label_calibration import \
    ThesauriLabelCalibrator, NotInitializedException
import tests.features.label_calibration.test_thesauri_label_calibration.common\
    as c


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


def test_fit_with_uninitialized_transformer_raises_exc(
        uninitialized_transformer, X
):
    with pytest.raises(NotInitializedException):
        ThesauriLabelCalibrator(uninitialized_transformer).fit(
            X, [[c.CONCEPT_x0]] * 2
        )


def test_predict_without_fit_raises_exc(calibrator, X):
    with pytest.raises(NotFittedError):
        calibrator.predict(X)


def test_predict(calibrator, X, y):
    calibrator.fit(X, y)
    assert (calibrator.predict(X) == np.array([[0, 0], [1, 1]])).all()
