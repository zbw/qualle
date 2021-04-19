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

from qualle.models import Scores


class ConfidenceFeatures(BaseEstimator, TransformerMixin):
    """Features based on aggregating concept level confidence scores."""

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: List[Scores]):
        _min = [np.min(row) if row else 0.0 for row in X]
        _mean = [np.mean(row) if row else 0.0 for row in X]
        _median = [np.median(row) if row else 0.0 for row in X]
        _prod = [np.prod(row) if row else 0.0 for row in X]
        return np.column_stack([_min, _mean, _median, _prod])
