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

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict

from qualle.label_calibration import LabelCalibrator
from qualle.models import TrainData, PredictData
from qualle.quality_estimation import RecallPredictor, RecallPredictorInput
from qualle.utils import recall


class QualityEstimationPipeline:

    def __init__(self):
        # TODO: Make regressors configurable
        self._lc = LabelCalibrator(
            ExtraTreesRegressor(n_estimators=10, min_samples_leaf=20)
        )
        self._rp = RecallPredictor(ExtraTreesRegressor())
        self._train_data_last_run = {}

    def train(self, data: TrainData):
        no_of_true_labels = np.array(list(map(len, data.true_concepts)))
        label_calibration = cross_val_predict(
            self._lc, data.docs, no_of_true_labels
        )
        self._lc.fit(data.docs, no_of_true_labels)

        no_of_pred_labels = np.array(list(map(len, data.predicted_concepts)))
        rp_input = RecallPredictorInput(
            no_of_pred_labels=no_of_pred_labels,
            label_calibration=label_calibration
        )
        true_recall = recall(data.true_concepts, data.predicted_concepts)
        self._rp.fit(rp_input, true_recall)

        self._train_data_last_run = dict(
            label_calibration=label_calibration,
            no_of_pred_labels=no_of_pred_labels,
            true_recall=true_recall
        )

    def reset_and_fit_recall_predictor(self, rp: RecallPredictor):
        self._rp = rp

        rp_input = RecallPredictorInput(
            no_of_pred_labels=self._train_data_last_run['no_of_pred_labels'],
            label_calibration=self._train_data_last_run['label_calibration']
        )
        self._rp.fit(rp_input, self._train_data_last_run['true_recall'])

    def predict(self, data: PredictData) -> List[float]:
        label_calibration = self._lc.predict(data.docs)
        no_of_pred_labels = np.array(list(map(len, data.predicted_concepts)))
        rp_input = RecallPredictorInput(
            no_of_pred_labels=no_of_pred_labels,
            label_calibration=label_calibration
        )
        return self._rp.predict(rp_input)
