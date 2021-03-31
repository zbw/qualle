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
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

from qualle.label_calibration.base import LabelCalibrator
from qualle.models import LabelCalibrationData, Documents, Concepts


def transform_to_label_count(X: List[Concepts]) -> np.array:
    return np.array(list(map(len, X)))


class SimpleLabelCalibrator(BaseEstimator, RegressorMixin):

    def __init__(
            self, regressor=ExtraTreesRegressor(
                n_estimators=10, min_samples_leaf=20
            )
    ):
        self.regressor = regressor

    def fit(self, X: Documents, y: List[Concepts]):
        self.calibrator_ = LabelCalibrator(self.regressor)
        y_transformed = transform_to_label_count(y)
        self.calibrator_.fit(X, y_transformed)
        return self

    def predict(self, X: Documents):
        check_is_fitted(self)
        return self.calibrator_.predict(X)


class SimpleLabelCalibrationFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: LabelCalibrationData):
        no_of_pred_labels = transform_to_label_count(X.predicted_concepts)
        rows = len(X.predicted_no_of_concepts)
        data = np.zeros((rows, 2))
        data[:, 0] = X.predicted_no_of_concepts
        data[:, 1] = X.predicted_no_of_concepts - no_of_pred_labels
        return data
