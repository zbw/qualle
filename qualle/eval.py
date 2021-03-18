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
import logging
from collections import OrderedDict
from random import random
from typing import Tuple, Generator, List, Dict

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from qualle.models import TrainData, PredictData
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor
from qualle.utils import recall, train_input_from_tsv

qe_estimators = [
    LinearRegression(), DecisionTreeRegressor(),
    ensemble.GradientBoostingRegressor(),
    ensemble.GradientBoostingRegressor(n_estimators=50, max_depth=4),
    ensemble.AdaBoostRegressor(), ensemble.ExtraTreesRegressor()
]


# TODO: add test
class Evaluator:

    def __init__(self, train_file: str, out_dir: str):
        self.train_file = train_file
        self.out_dir = out_dir
        self.scores = None

    def evaluate(self):
        df_scores = pd.DataFrame()
        for fold_scores in self.evaluate_folds():
            df_scores = df_scores.append(pd.DataFrame(fold_scores))
            df_scores.to_csv(self.out_dir + "/qe_avg_metrics.tsv", sep='\t',
                             index=False)

    def evaluate_folds(self):
        t_input = train_input_from_tsv(self.train_file)
        for fold_idx, (train, eval) in enumerate(gen_k_folds(t_input)):
            logging.warn('Evaluating fold %d', fold_idx)

            qe_p = None
            for qe_e in qe_estimators:
                rp = RecallPredictor(qe_e)
                if qe_p is None:
                    qe_p = QualityEstimationPipeline(rp)
                    qe_p.train(train)
                else:
                    qe_p.reset_and_fit_recall_predictor(rp)

                pred_recall = qe_p.predict(
                    PredictData(docs=eval.docs,
                                predicted_concepts=eval.predicted_concepts)
                )
                true_recall = recall(eval.true_concepts,
                                     eval.predicted_concepts)
                scores_for_qe_e = OrderedDict()

                scores_for_qe_e['fold'] = fold_idx
                scores_for_qe_e['qe_estimator'] = str(qe_e)
                scores_for_qe_e.update(scores(true_recall, pred_recall))
                yield scores_for_qe_e


def scores(true_recall: List[float], pred_recall: List[float]) -> Dict:
    scores = dict()
    scores['explained_variance_score'] = [explained_variance_score(
        true_recall, pred_recall
    )]
    scores['mean_squared_error'] = [
        mean_squared_error(true_recall, pred_recall)
    ]
    scores['correlation_coefficient'] = [np.corrcoef(true_recall, pred_recall)[
        0, 1]]
    return scores


def gen_k_folds(t_input: TrainData) -> Generator[
        Tuple[TrainData, TrainData], None, None]:
    kf = KFold(n_splits=5, shuffle=True, random_state=4)
    for (train_idx, eval_idx) in kf.split(t_input.docs):
        train = TrainData(
            docs=[t_input.docs[i] for i in train_idx],
            predicted_concepts=[
                t_input.predicted_concepts[i] for i in train_idx],
            true_concepts=[t_input.true_concepts[i] for i in train_idx]
        )
        eval = TrainData(
            docs=[t_input.docs[i] for i in eval_idx],
            predicted_concepts=[t_input.predicted_concepts[i] for i in
                                eval_idx],
            true_concepts=[t_input.true_concepts[i] for i in eval_idx]
        )
        yield train, eval
