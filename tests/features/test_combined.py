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
from typing import List

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qualle.features.base import Features
from qualle.features.combined import CombinedFeatures


class FeatureA(Features):

    def transform(self, X: List[List[str]]):
        return np.array(list(map(lambda x: [x], map(len, X))))


class FeatureB(Features):

    def transform(self, X: List[List[float]]):
        return np.array(list(map(lambda x: [x], map(sum, X))))


@pytest.fixture
def X():
    return {
        FeatureA: [['x0', 'x1'], ['x2']],
        FeatureB: [[3, 4], [5]]
    }


@pytest.fixture
def combined_features():
    return CombinedFeatures([FeatureA(), FeatureB()])


def test_combined_fit_combines_features(combined_features, X, mocker):
    spies = []
    for f in combined_features.features:
        spies.append(mocker.spy(f, 'fit'))
    combined_features.fit(X)
    for i, spy in enumerate(spies):
        spy.assert_called_once_with(X[combined_features.features[i].__class__])


def test_transform_without_fit_raises_exc(combined_features, X):
    with pytest.raises(NotFittedError):
        combined_features.transform(X)


def test_transform_hstacks_result(combined_features, X):
    combined_features.fit(X)
    transformed = combined_features.transform(X)
    assert type(transformed) == np.ndarray
    assert transformed.shape == (2, 2)
    assert (transformed == [[2, 7], [1, 5]]).all()
