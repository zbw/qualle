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
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError

from qualle.features.text import TextFeatures


@pytest.fixture
def data():
    return [
        'Oneword',
        '2word speci?l',
    ]


def test_fit_fits_underlying_features(data, mocker):
    f_mock = mocker.Mock()
    m = mocker.Mock(return_value=f_mock)
    mocker.patch('qualle.features.text.mk_text_features', m)
    cf = TextFeatures()
    cf.fit(data)
    assert hasattr(cf, 'features_')
    assert cf.features_ == f_mock
    f_mock.fit.assert_called_once_with(data)


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
