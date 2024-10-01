#  Copyright 2021-2024 ZBW – Leibniz Information Centre for Economics
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

from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrationFeatures,
)
from qualle.models import LabelCalibrationData


@pytest.fixture
def features():
    return SimpleLabelCalibrationFeatures()


def test_transform(features):
    data = LabelCalibrationData(
        predicted_labels=[["c0"], ["c0", "c1"]], predicted_no_of_labels=np.array([1, 4])
    )
    assert (features.transform(data) == [[1, 0], [4, 2]]).all()
