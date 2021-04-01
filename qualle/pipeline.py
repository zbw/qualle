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

from sklearn.model_selection import cross_val_predict

from qualle.features.label_calibration.base import AbstractLabelCalibrator
from qualle.models import TrainData, PredictData, LabelCalibrationData
from qualle.quality_estimation import RecallPredictor
from qualle.utils import recall


class QualityEstimationPipeline:

    def __init__(
            self,
            label_calibrator: AbstractLabelCalibrator,
            rp: RecallPredictor
    ):
        self._label_calibrator = label_calibrator
        self._rp = rp

    def train(self, data: TrainData):
        predicted_no_of_labels = cross_val_predict(
            self._label_calibrator, data.docs, data.true_concepts
        )
        self._label_calibrator.fit(data.docs, data.true_concepts)

        label_calibration_data = LabelCalibrationData(
            predicted_concepts=data.predicted_concepts,
            predicted_no_of_concepts=predicted_no_of_labels
        )
        true_recall = recall(data.true_concepts, data.predicted_concepts)
        self._rp.fit(label_calibration_data, true_recall)

    def predict(self, data: PredictData) -> List[float]:
        predicted_no_of_labels = self._label_calibrator.predict(data.docs)
        label_calibration_data = LabelCalibrationData(
            predicted_concepts=data.predicted_concepts,
            predicted_no_of_concepts=predicted_no_of_labels
        )
        return self._rp.predict(label_calibration_data)
