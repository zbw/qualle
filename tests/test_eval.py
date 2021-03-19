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

from qualle import evaluate
from qualle.evaluate import Evaluator, scores


@pytest.fixture
def train_tsv():
    return '\n'.join(
        [f'Title{i}\tpred_concept{i}1,pred_concept{i}2\t'
         f'true_concept{i}1,true_concept{i}2' for i in range(20)]
    )


@pytest.fixture
def evaluator(train_tsv, mocker):
    m = mocker.mock_open(read_data=train_tsv)
    mocker.patch('qualle.utils.open', m)
    return Evaluator('', '')


def test_evaluate_returns_scores_for_each_estimator(evaluator):
    l_of_scores = evaluator.evaluate()

    assert set(str(qe_est) for qe_est in evaluate.QE_ESTIMATORS) == set([
        s['qe_estimator'] for s in l_of_scores
    ])
    assert {'qe_estimator', 'explained_variance_score', 'mean_squared_error',
            'correlation_coefficient'} == set(
        key for s in l_of_scores for key in s.keys())


def test_evaluate_trains_fully_only_once(evaluator, mocker):
    spy = mocker.spy(evaluator._qe_p, 'train')

    evaluator.evaluate()

    assert spy.call_count == 1


def test_evaluate_resets_recall_predictor_for_each_nonfirst_estimator(
        evaluator, mocker
):
    spy = mocker.spy(evaluator._qe_p, 'reset_and_fit_recall_predictor')

    evaluator.evaluate()

    assert spy.call_count == len(evaluate.QE_ESTIMATORS) - 1
    assert set(map(lambda x: x[0][0].regressor, spy.call_args_list)) == set(
        evaluate.QE_ESTIMATORS[1:])


def test_scores():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    s = scores(y_true, y_pred)

    assert s['explained_variance_score'] == 0.9571734475374732
    assert s['mean_squared_error'] == 0.375
    assert s['correlation_coefficient'] == 0.9848696184482703
