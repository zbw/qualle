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

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from qualle.features.base import Features


class QualityEstimator(BaseEstimator, RegressorMixin):
    """Regressor which uses a pipeline with given features and regressor."""

    def __init__(
            self,
            regressor: RegressorMixin,
            features: Features
    ):
        self.regressor = regressor
        self.features = features

    def fit(self, X, y):
        self.pipeline_ = Pipeline([
            ("features", self.features),
            ("regressor", self.regressor)
        ])
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict(X)
