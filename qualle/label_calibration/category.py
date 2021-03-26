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
from sklearn.base import BaseEstimator

from qualle.label_calibration.base import LabelCalibrator


class MultiCategoryLabelCalibrator(BaseEstimator):
    """Label calibrator for multiple distinct categories.
    E.g. to predict the no of labels for different thesauri."""

    def __init__(self, regressor, no_categories: int):
        if no_categories < 1:
            raise ValueError('Number of categories must be greater 0')
        self._calibrators = [
            LabelCalibrator(regressor) for _ in range(no_categories)
        ]

    def fit(self, X: List[str], y: np.array):
        """
        :param X: list of content, e.g. doc titles
        :param y: 2-dim array where each row corresponds to no of labels for
            respective category
        :return: self
        """
        for i, c in enumerate(self._calibrators):
            c.fit(X, y[i])
        return self

    def predict(self, X: List[str]):
        return np.vstack([c.predict(X) for c in self._calibrators])
