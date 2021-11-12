#  Copyright 2021 ZBW  – Leibniz Information Centre for Economics
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
from stwfsapy.text_features import mk_text_features

from qualle.label_calibration.simple import LabelCalibrator
from tests.common import DummyRegressor


@pytest.fixture
def calibrator():
    return LabelCalibrator(DummyRegressor())


def test_lc_predict(calibrator, X):
    calibrator.fit(X, [3, 5])
    assert np.array_equal(calibrator.predict(X), [0, 1])


def test_lc_predict_without_fit_raises_exc(calibrator, X):
    with pytest.raises(NotFittedError):
        calibrator.predict(X)


def test_lc_fit_fits_regressor_with_txt_features(calibrator, X, mocker):
    y = [3, 5]
    txt_features = mk_text_features().fit(X)
    X_transformed = txt_features.transform(X)

    spy = mocker.spy(calibrator.regressor, 'fit')
    calibrator.fit(X, y)
    spy.assert_called_once()
    assert (spy.call_args[0][0].toarray() == X_transformed.toarray()).all()
    assert spy.call_args[0][1] == y
