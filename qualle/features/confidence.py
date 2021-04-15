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
from sklearn.base import TransformerMixin, BaseEstimator

Scores = List[float]


class ConfidenceFeatures(BaseEstimator, TransformerMixin):
    """Features based on aggregating concept level confidence scores."""

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: List[Scores]):
        array = np.array(X)
        return np.column_stack([
            np.min(array, axis=1), np.mean(array, axis=1),
            np.median(array, axis=1), np.prod(array, axis=1)
        ])
