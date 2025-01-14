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
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from qualle.features.base import Features
from qualle.features.label_calibration.base import AbstractLabelCalibrator
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


class SimpleLabelCalibrationFeatures(Features):
    def transform(self, X: LabelCalibrationData):
        no_of_pred_labels = transform_to_label_count(X.predicted_labels)
        rows = len(X.predicted_no_of_labels)
        data = np.zeros((rows, 2))
        data[:, 0] = X.predicted_no_of_labels
        data[:, 1] = X.predicted_no_of_labels - no_of_pred_labels
        return data
