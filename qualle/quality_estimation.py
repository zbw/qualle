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
from dataclasses import dataclass

import numpy as np
from sklearn import ensemble
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


@dataclass
class RecallPredictorInput:
    label_calibration: np.array
    no_of_pred_labels: np.array


class RecallPredictor(BaseEstimator, RegressorMixin):

    def __init__(self, regressor=ensemble.AdaBoostRegressor()):
        self.regressor = regressor

    def fit(self, X: RecallPredictorInput, y):
        self.pipeline_ = Pipeline([
            ("features", LabelCalibrationFeatures()),
            ("regressor", self.regressor)
        ])
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X: RecallPredictorInput):
        check_is_fitted(self)
        return self.pipeline_.predict(X)


class LabelCalibrationFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: RecallPredictorInput):
        rows = len(X.label_calibration)
        data = np.zeros((rows, 2))
        data[:, 0] = X.label_calibration
        data[:, 1] = X.label_calibration - X.no_of_pred_labels
        return data
