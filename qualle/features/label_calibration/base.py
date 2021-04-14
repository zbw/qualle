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

from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin

from qualle.models import LabelCalibrationData, Documents, Labels


class AbstractLabelCalibrator(BaseEstimator, RegressorMixin):

    def fit(self, X: Documents, y: List[Labels]):
        pass  # pragma: no cover

    def predict(self, X: Documents):
        pass  # pragma: no cover


class AbstractLabelCalibrationFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        pass  # pragma: no cover

    def transform(self, X: LabelCalibrationData):
        pass  # pragma: no cover
