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
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError

from qualle.features.text import TextFeatures


@pytest.fixture
def data():
    return [
        'Oneword',
        '2word speci?l',
    ]


def test_transform_without_fit_raises_exc(data):
    with pytest.raises(NotFittedError):
        TextFeatures().transform(data)


def test_transform_computes_all_features(data):
    cf = TextFeatures()
    cf.fit(data)
    features = cf.transform(data)
    assert sp.issparse(features)
    features_array = features.toarray()

    vect_feat = features_array[:, 0:3]
    n_char_feat = features_array[:, 3]
    n_word_feat = features_array[:, 4]
    n_special_feat = features_array[:, 5]
    n_upper_feat = features_array[:, 6]
    n_digit_feat = features_array[:, 7]

    # we cannot rely on the order of the matrix produced by the vectorizer,
    # therefore we assert nonzero indices of rows are disjoint because words
    # are disjoint
    assert (
        set(np.flatnonzero(vect_feat[0])) & set(np.flatnonzero(vect_feat[1]))
        == set()
    )
    assert (n_char_feat == [7, 13]).all()
    assert (n_word_feat == [0, 1]).all()
    assert (n_special_feat == [0, 1]).all()
    assert (n_upper_feat == [1, 0]).all()
    assert (n_digit_feat == [0, 1]).all()
