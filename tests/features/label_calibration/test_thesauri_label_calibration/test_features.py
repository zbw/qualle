#  Copyright 2021-2023 ZBW – Leibniz Information Centre for Economics
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

from qualle.features.label_calibration.thesauri_label_calibration import \
    ThesauriLabelCalibrationFeatures, NotInitializedException
from qualle.models import LabelCalibrationData

import tests.features.label_calibration.test_thesauri_label_calibration.common\
    as c


@pytest.fixture
def features(transformer):
    return ThesauriLabelCalibrationFeatures(transformer)


def test_transform(features):
    data = LabelCalibrationData(
        predicted_labels=[[c.CONCEPT_x0], [c.CONCEPT_x1, c.CONCEPT_x2]],
        predicted_no_of_labels=np.array([[1, 2], [0, 3]])
    )
    assert (features.transform(data) == [
        [1, 2, 0, 2], [0, 3, -2, 1]
    ]).all()


def test_transform_with_uninitialized_underlying_tranformer_raises_exc(
        uninitialized_transformer
):
    with pytest.raises(NotInitializedException):
        ThesauriLabelCalibrationFeatures(uninitialized_transformer).transform(
            LabelCalibrationData(
                predicted_labels=[c.CONCEPT_x0],
                predicted_no_of_labels=np.array([[1]])
            )
        )
