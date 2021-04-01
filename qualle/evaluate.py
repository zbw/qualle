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
from typing import List, Dict

import numpy as np
from sklearn.metrics import explained_variance_score, mean_squared_error

from qualle.models import PredictData, TrainData
from qualle.pipeline import QualityEstimationPipeline
from qualle.utils import recall


class Evaluator:

    def __init__(self, test_data: TrainData, qe_p: QualityEstimationPipeline):
        self._test_data = test_data
        self._qe_p = qe_p

    def evaluate(self) -> Dict:
        pred_recall = self._qe_p.predict(
            PredictData(docs=self._test_data.docs,
                        predicted_labels=self._test_data.predicted_labels)
        )
        true_recall = recall(self._test_data.true_labels,
                             self._test_data.predicted_labels)

        return scores(true_recall, pred_recall)


def scores(true_recall: List[float], pred_recall: List[float]) -> Dict:
    scores = dict()
    scores['explained_variance_score'] = explained_variance_score(
        true_recall, pred_recall
    )
    scores['mean_squared_error'] = mean_squared_error(
        true_recall, pred_recall)

    scores['correlation_coefficient'] = np.corrcoef(true_recall, pred_recall)[
        0, 1]
    return scores
