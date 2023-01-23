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
from scipy.sparse import issparse

from qualle.features.label_calibration.thesauri_label_calibration import (
    NotInitializedException,
)
from tests.features.label_calibration.test_thesauri_label_calibration import common as c


def test_transform_without_init_raises_exc(uninitialized_transformer):
    with pytest.raises(NotInitializedException):
        uninitialized_transformer.transform([])


def test_transform_returns_count_matrix(transformer):
    assert (
        transformer.transform(
            [[c.CONCEPT_x0, c.CONCEPT_INVALID], [c.CONCEPT_x0, c.CONCEPT_x2]]
        )
        == np.array([[1, 0], [2, 1]])
    ).all()


def test_transform_returns_sparse_count_matrix(transformer):
    transformer._use_sparse_count_matrix = True
    count_matrix = transformer.transform(
        [[c.CONCEPT_x0, c.CONCEPT_INVALID], [c.CONCEPT_x0, c.CONCEPT_x2]]
    )
    assert issparse(count_matrix)
    assert (count_matrix.toarray() == np.array([[1, 0], [2, 1]])).all()
