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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

from qualle.label_calibration.base import LabelCalibrator


class MultiCategoryLabelCalibrator(BaseEstimator, RegressorMixin):
    """Label calibrator for multiple distinct categories.
    E.g. to predict the no of labels for different thesauri."""

    def __init__(self, regressor_class=ExtraTreesRegressor):
        self.regressor_class = regressor_class

    def fit(self, X: List[str], y: np.array):
        """
        :param X: list of content, e.g. doc titles
        :param y: 2-dim array where each column corresponds to no of labels for
            respective category
        :return: self
        """
        try:
            no_categories = y.shape[1]
        except IndexError:
            raise ValueError('Number of categories must be greater 0')

        self.calibrators_ = [
            LabelCalibrator(self.regressor_class())
            for _ in range(no_categories)
        ]

        for i, c in enumerate(self.calibrators_):
            c.fit(X, y[:, i])
        return self

    def predict(self, X: List[str]):
        check_is_fitted(self)
        return np.vstack([c.predict(X) for c in self.calibrators_])
