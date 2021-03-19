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
from typing import Generator, List, Dict

import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from qualle.models import PredictData
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor
from qualle.utils import recall, train_input_from_tsv

# TODO: make configurable
QE_ESTIMATORS = [
    LinearRegression(), DecisionTreeRegressor(),
    ensemble.GradientBoostingRegressor(),
    ensemble.GradientBoostingRegressor(n_estimators=50, max_depth=4),
    ensemble.AdaBoostRegressor(), ensemble.ExtraTreesRegressor()
]


class Evaluator:

    def __init__(self, train_file: str, eval_file: str):
        self._train_file = train_file
        self._eval_file = eval_file
        self._qe_p = QualityEstimationPipeline(RecallPredictor(
            QE_ESTIMATORS[0]))

    def evaluate(self) -> List[Dict]:
        return [s for s in self._evaluate_estimators()]

    def _evaluate_estimators(self) -> Generator[Dict, None, None]:
        t_input = train_input_from_tsv(self._train_file)
        e_input = train_input_from_tsv(self._eval_file)

        for i, qe_e in enumerate(QE_ESTIMATORS):
            if i == 0:
                self._qe_p.train(t_input)
            else:
                rp = RecallPredictor(qe_e)
                self._qe_p.reset_and_fit_recall_predictor(rp)

            pred_recall = self._qe_p.predict(
                PredictData(docs=e_input.docs,
                            predicted_concepts=e_input.predicted_concepts)
            )
            true_recall = recall(e_input.true_concepts,
                                 e_input.predicted_concepts)

            scores_for_qe_e = dict(qe_estimator=str(qe_e))
            scores_for_qe_e.update(scores(true_recall, pred_recall))
            yield scores_for_qe_e


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
