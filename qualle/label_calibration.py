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

from sklearn.pipeline import Pipeline

from stwfsapy.text_features import mk_text_features


class LabelCalibrator:

    def __init__(self, regressor):
        features = mk_text_features()
        self._pipeline = Pipeline(
            [("features", features), ("regressor", regressor)]
        )

    def fit(self, X: List[str], y):
        self._pipeline.fit(X, y)

    def predict(self, X: List[str]):
        return self._pipeline.predict(X)
