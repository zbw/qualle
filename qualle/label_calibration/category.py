#  Copyright 2021-2025 ZBW – Leibniz Information Centre for Economics
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
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

from qualle.label_calibration.simple import LabelCalibrator
from qualle.models import Matrix


class MultiCategoryLabelCalibrator(BaseEstimator, RegressorMixin):
    """Label calibrator for multiple distinct categories.
    E.g. to predict the no of labels for different thesauri."""

    def __init__(self, regressor_class=ExtraTreesRegressor, regressor_params=None):
        self.regressor_class = regressor_class
        self.regressor_params = regressor_params or {}

    def fit(self, X: List[str], y: Matrix):
        """
        :param X: list of content, e.g. doc titles
        :param y: matrix where each column corresponds to no of labels for
            respective category
        :return: self
        """
        try:
            no_categories = y.shape[1]
        except IndexError:
            raise ValueError("Number of categories must be greater 0")

        self.calibrators_ = [
            LabelCalibrator(self.regressor_class(**self.regressor_params))
            for _ in range(no_categories)
        ]

        y_is_sparse = issparse(y)
        for i, c in enumerate(self.calibrators_):
            if y_is_sparse:
                col = y.getcol(i).toarray().flatten()
            else:
                col = y[:, i]
            c.fit(X, col)
        return self

    def predict(self, X: List[str]):
        check_is_fitted(self)
        return np.column_stack([c.predict(X) for c in self.calibrators_])
