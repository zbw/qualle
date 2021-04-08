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
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from qualle.features.label_calibration.base import AbstractLabelCalibrator, \
    AbstractLabelCalibrationFeatures
from qualle.label_calibration.simple import LabelCalibrator
from qualle.models import LabelCalibrationData, Documents, Labels


def transform_to_label_count(X: List[Labels]) -> np.array:
    return np.array(list(map(len, X)))


class SimpleLabelCalibrator(AbstractLabelCalibrator):

    def __init__(self, regressor: RegressorMixin):
        self.regressor = regressor

    def fit(self, X: Documents, y: List[Labels]):
        self.calibrator_ = LabelCalibrator(self.regressor)
        y_transformed = transform_to_label_count(y)
        self.calibrator_.fit(X, y_transformed)
        return self

    def predict(self, X: Documents):
        check_is_fitted(self)
        return self.calibrator_.predict(X)


class SimpleLabelCalibrationFeatures(AbstractLabelCalibrationFeatures):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: LabelCalibrationData):
        no_of_pred_labels = transform_to_label_count(X.predicted_labels)
        rows = len(X.predicted_no_of_labels)
        data = np.zeros((rows, 2))
        data[:, 0] = X.predicted_no_of_labels
        data[:, 1] = X.predicted_no_of_labels - no_of_pred_labels
        return data
