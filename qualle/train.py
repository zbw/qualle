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


from qualle.features.label_calibration.base import AbstractLabelCalibrator
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor


class Trainer:

    def __init__(
            self, train_data,
            label_calibrator: AbstractLabelCalibrator,
            recall_predictor: RecallPredictor,
            should_debug=False
    ):
        self._qe_p = QualityEstimationPipeline(
            recall_predictor=recall_predictor,
            label_calibrator=label_calibrator,
            should_debug=should_debug,
            # Can't use lambda because of pickle
            features_data_mapper=features_data_mapper
        )
        self._train_data = train_data

    def train(self) -> QualityEstimationPipeline:
        self._qe_p.train(self._train_data)

        return self._qe_p


def features_data_mapper(_, label_calibration_data):
    return label_calibration_data
