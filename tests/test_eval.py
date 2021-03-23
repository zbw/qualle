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
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from qualle.evaluate import Evaluator, scores
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor


@pytest.fixture
def evaluator(train_data):
    qe_p = QualityEstimationPipeline(RecallPredictor(ExtraTreesRegressor()))
    qe_p.train(train_data)
    return Evaluator(train_data, qe_p)


def test_evaluate_returns_scores(evaluator):
    scores = evaluator.evaluate()

    assert {'explained_variance_score', 'mean_squared_error',
            'correlation_coefficient'} == set(scores.keys())


def test_scores():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    s = scores(y_true, y_pred)

    assert s['explained_variance_score'] == 0.9571734475374732
    assert s['mean_squared_error'] == 0.375
    assert s['correlation_coefficient'] == 0.9848696184482703
