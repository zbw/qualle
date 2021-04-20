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
from qualle.features.confidence import ConfidenceFeatures


@pytest.fixture
def data():
    return [
        [0] * 2,
        [1] * 5,
        list(range(3)),
        list(range(3))[::-1],
        [1, 2, 4]
    ]


def test_transform_computes_all_features(data):
    cf = ConfidenceFeatures()
    features = cf.transform(data)
    assert type(features) == np.ndarray
    assert (features == np.vstack([
        [0] * 4,
        [1] * 4,
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 7 / 3, 2, 8]
    ])).all()


def test_transform_empty_row_gets_zero_value_as_default():
    cf = ConfidenceFeatures()

    features = cf.transform([[], [1] * 5])
    assert type(features) == np.ndarray
    assert (features == np.vstack([
        [0] * 4,
        [1] * 4,
    ])).all()
