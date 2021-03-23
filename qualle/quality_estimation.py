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
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


@dataclass
class RecallPredictorInput:
    label_calibration: np.array
    no_of_pred_labels: np.array


class RecallPredictor(BaseEstimator):

    def __init__(self, regressor):
        self.regressor = regressor
        self.pipeline = Pipeline([
            ("features", LabelCalibrationFeatures()),
            ("regressor", regressor)
        ])

    def fit(self, X: RecallPredictorInput, y):
        self.pipeline.fit(X, y)

    def predict(self, X: RecallPredictorInput):
        return self.pipeline.predict(X)


class LabelCalibrationFeatures():

    def fit(self, X, y=None):
        return self

    def transform(self, X: RecallPredictorInput):
        rows = len(X.label_calibration)
        data = np.zeros((rows, 2))
        data[:, 0] = X.label_calibration
        data[:, 1] = X.label_calibration - X.no_of_pred_labels
        return data
