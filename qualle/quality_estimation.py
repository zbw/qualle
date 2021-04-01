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

from sklearn import ensemble
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from qualle.features.label_calibration.base import \
    AbstractLabelCalibrationFeatures
from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrationFeatures
from qualle.models import LabelCalibrationData


class RecallPredictor(BaseEstimator, RegressorMixin):

    def __init__(
            self,
            regressor=ensemble.AdaBoostRegressor(),
            label_calibration_features: AbstractLabelCalibrationFeatures =
            SimpleLabelCalibrationFeatures()
    ):
        self.regressor = regressor
        self.label_calibration_features = label_calibration_features

    def fit(self, X: LabelCalibrationData, y):
        self.pipeline_ = Pipeline([
            ("features", self.label_calibration_features),
            ("regressor", self.regressor)
        ])
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X: LabelCalibrationData):
        check_is_fitted(self)
        return self.pipeline_.predict(X)
