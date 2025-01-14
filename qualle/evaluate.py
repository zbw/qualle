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
from typing import List, Dict

import numpy as np
from sklearn.metrics import explained_variance_score, mean_squared_error

from qualle.models import TrainData
from qualle.pipeline import QualityEstimationPipeline
from qualle.utils import recall


class Evaluator:
    def __init__(self, test_data: TrainData, qe_p: QualityEstimationPipeline):
        self._test_data = test_data
        self._qe_p = qe_p

    def evaluate(self) -> Dict:
        predict_data = self._test_data.predict_data
        pred_recall = self._qe_p.predict(predict_data)
        true_recall = recall(self._test_data.true_labels, predict_data.predicted_labels)

        return scores(true_recall, pred_recall)


def scores(true_recall: List[float], pred_recall: List[float]) -> Dict:
    scores = dict()
    scores["explained_variance_score"] = explained_variance_score(
        true_recall, pred_recall
    )
    scores["mean_squared_error"] = mean_squared_error(true_recall, pred_recall)

    scores["correlation_coefficient"] = np.corrcoef(true_recall, pred_recall)[0, 1]
    return scores
