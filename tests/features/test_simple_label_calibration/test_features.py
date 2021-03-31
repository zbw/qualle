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

from qualle.features.simple_label_calibration import \
    SimpleLabelCalibrationFeatures
from qualle.models import LabelCalibrationData


@pytest.fixture
def features():
    return SimpleLabelCalibrationFeatures()


def test_transform(features):
    data = LabelCalibrationData(
        predicted_concepts=[['c0'], ['c0', 'c1']],
        predicted_no_of_concepts=np.array([1, 4])
    )
    assert (features.transform(data) == [[1, 0], [4, 2]]).all()
