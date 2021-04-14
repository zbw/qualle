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

from qualle.features.label_calibration.thesauri_label_calibration import \
    ThesauriLabelCalibrationFeatures
from qualle.models import LabelCalibrationData

import tests.features.label_calibration.test_thesauri_label_calibration.common\
    as c


@pytest.fixture
def features(transformer):
    transformer.fit()
    return ThesauriLabelCalibrationFeatures(transformer)


def test_transform(features):
    data = LabelCalibrationData(
        predicted_labels=[[c.CONCEPT_x0], [c.CONCEPT_x1, c.CONCEPT_x2]],
        predicted_no_of_labels=np.array([[1, 2], [0, 3]])
    )
    assert (features.transform(data) == [
        [1, 2, 0, 2], [0, 3, -2, 1]
    ]).all()


def test_transform_with_unfitted_underlying_tranformer_raises_exc(transformer):
    with pytest.raises(NotFittedError):
        ThesauriLabelCalibrationFeatures(transformer).transform(
            LabelCalibrationData(
                predicted_labels=[c.CONCEPT_x0],
                predicted_no_of_labels=np.array([[1]])
            )
        )
