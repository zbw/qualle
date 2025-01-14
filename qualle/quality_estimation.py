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

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from qualle.features.base import Features


class QualityEstimator(BaseEstimator, RegressorMixin):
    """Regressor which uses a pipeline with given features and regressor."""

    def __init__(self, regressor: RegressorMixin, features: Features):
        self.regressor = regressor
        self.features = features

    def fit(self, X, y):
        self.pipeline_ = Pipeline(
            [("features", self.features), ("regressor", self.regressor)]
        )
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict(X)
