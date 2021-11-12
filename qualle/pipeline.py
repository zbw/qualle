#  Copyright 2021 ZBW  – Leibniz Information Centre for Economics
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
from contextlib import contextmanager
from typing import List, Callable, Any

from sklearn.model_selection import cross_val_predict

from qualle.features.label_calibration.base import AbstractLabelCalibrator
from qualle.models import TrainData, PredictData, LabelCalibrationData
from qualle.quality_estimation import QualityEstimator
from qualle.utils import recall, get_logger, timeit


class QualityEstimationPipeline:

    def __init__(
            self,
            label_calibrator: AbstractLabelCalibrator,
            recall_predictor: QualityEstimator,
            features_data_mapper: Callable[
                [PredictData, LabelCalibrationData], Any
            ],
            should_debug=False
    ):
        self._label_calibrator = label_calibrator
        self._recall_predictor = recall_predictor
        self._features_data_mapper = features_data_mapper
        self._should_debug = should_debug
        self._logger = get_logger()

    def train(self, data: TrainData):
        predict_data = data.predict_data
        with self._debug('cross_val_predict with label calibrator'):
            predicted_no_of_labels = cross_val_predict(
                self._label_calibrator, predict_data.docs, data.true_labels
            )

        with self._debug('label calibrator fit'):
            self._label_calibrator.fit(predict_data.docs, data.true_labels)

        label_calibration_data = LabelCalibrationData(
            predicted_labels=predict_data.predicted_labels,
            predicted_no_of_labels=predicted_no_of_labels
        )
        features_data = self._features_data_mapper(
            predict_data, label_calibration_data
        )
        with self._debug('recall computation'):
            true_recall = recall(
                data.true_labels, predict_data.predicted_labels
            )

        with self._debug('RecallPredictor fit'):
            self._recall_predictor.fit(features_data, true_recall)

    def predict(self, data: PredictData) -> List[float]:
        predicted_no_of_labels = self._label_calibrator.predict(data.docs)
        label_calibration_data = LabelCalibrationData(
            predicted_labels=data.predicted_labels,
            predicted_no_of_labels=predicted_no_of_labels
        )
        features_data = self._features_data_mapper(
            data, label_calibration_data
        )
        return self._recall_predictor.predict(features_data)

    @contextmanager
    def _debug(self, method_name):
        if self._should_debug:
            with timeit() as t:
                yield
            self._logger.debug('Ran %s in %.4f seconds', method_name, t())
        else:
            yield

    def __str__(self):
        return f'{self._label_calibrator}\n{self._recall_predictor}'
