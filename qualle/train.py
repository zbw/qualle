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
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor


class Trainer:

    def __init__(self, train_data):
        # TODO: make regressor configurable
        self._qe_p = QualityEstimationPipeline(RecallPredictor(
            ensemble.AdaBoostRegressor())
        )
        self._train_data = train_data

    def train(self) -> QualityEstimationPipeline:
        self._qe_p.train(self._train_data)

        return self._qe_p
