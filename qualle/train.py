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

from sklearn.ensemble import GradientBoostingRegressor

from qualle.features.label_calibration.base import AbstractLabelCalibrator, \
    AbstractLabelCalibrationFeatures
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor


class Trainer:

    def __init__(
            self, train_data,
            label_calibrator: AbstractLabelCalibrator,
            label_calibration_features: AbstractLabelCalibrationFeatures,
            should_debug=False
    ):
        # TODO: make regressor configurable
        self._qe_p = QualityEstimationPipeline(
            rp=RecallPredictor(
                regressor=GradientBoostingRegressor(
                    n_estimators=10, max_depth=8),
                label_calibration_features=label_calibration_features
            ),
            label_calibrator=label_calibrator,
            should_debug=should_debug
        )
        self._train_data = train_data

    def train(self) -> QualityEstimationPipeline:
        self._qe_p.train(self._train_data)

        return self._qe_p
