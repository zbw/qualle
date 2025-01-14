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
import pytest

import numpy as np
from qualle.features.confidence import ConfidenceFeatures


@pytest.fixture
def data():
    return [[0] * 2, [1] * 5, list(range(3)), list(range(3))[::-1], [1, 2, 4]]


def test_transform_computes_all_features(data):
    cf = ConfidenceFeatures()
    features = cf.transform(data)
    assert type(features) == np.ndarray
    assert (
        features
        == np.vstack([[0] * 4, [1] * 4, [0, 1, 1, 0], [0, 1, 1, 0], [1, 7 / 3, 2, 8]])
    ).all()


def test_transform_empty_row_gets_zero_value_as_default():
    cf = ConfidenceFeatures()

    features = cf.transform([[], [1] * 5])
    assert type(features) == np.ndarray
    assert (
        features
        == np.vstack(
            [
                [0] * 4,
                [1] * 4,
            ]
        )
    ).all()
