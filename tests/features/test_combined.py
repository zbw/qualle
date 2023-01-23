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
from typing import List

import numpy as np
import pytest
import scipy.sparse as sp
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
    return {FeatureA: [["x0", "x1"], ["x2"]], FeatureB: [[3, 4], [5]]}


@pytest.fixture
def combined_features():
    return CombinedFeatures([FeatureA(), FeatureB()])


def test_combined_fit_combines_features(combined_features, X, mocker):
    spies = []
    for f in combined_features.features:
        spies.append(mocker.spy(f, "fit"))
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


def test_transform_uses_sparse_hstack_if_any_feature_is_sparse(combined_features, X):
    class SparseFeature(Features):
        def transform(self, X: List[List[float]]):
            return sp.csr_matrix([[1, 0], [3, 5]])

    combined_features.set_params(features=[FeatureA(), FeatureB(), SparseFeature()])
    X[SparseFeature] = [[0], [1]]
    combined_features.fit(X)
    transformed = combined_features.transform(X)
    assert sp.issparse(transformed)
    assert (transformed.toarray() == [[2, 7, 1, 0], [1, 5, 3, 5]]).all()
