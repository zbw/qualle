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

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import NotFittedError

from qualle.models import TrainData, PredictData
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor


@pytest.fixture
def qp():
    return QualityEstimationPipeline(RecallPredictor(ExtraTreesRegressor()))


@pytest.fixture
def train_data():
    concepts = [['c'] for _ in range(5)]
    return TrainData(
        docs=[f'd{i}' for i in range(5)],
        true_concepts=concepts,
        predicted_concepts=concepts
    )


def test_train(qp, train_data, mocker):
    spy_lc = mocker.spy(qp._lc, 'fit')
    spy_rp = mocker.spy(qp._rp, 'fit')

    qp.train(train_data)

    actual_lc_docs = spy_lc.call_args[0][0]
    actual_lc_no_of_true_labels = spy_lc.call_args[0][1]

    assert actual_lc_docs == train_data.docs
    assert np.array_equal(actual_lc_no_of_true_labels, [1] * 5)

    actual_rp_input = spy_rp.call_args[0][0]
    actual_true_recall = spy_rp.call_args[0][1]

    # Because of how our input data is designed,
    # we can make following assertions
    only_ones = [1] * 5
    assert np.array_equal(actual_rp_input.no_of_pred_labels, only_ones)
    assert np.array_equal(actual_rp_input.label_calibration, only_ones)
    assert actual_true_recall == only_ones


def test_predict_wihout_train_raises_exc(qp):
    data = PredictData(
        docs=[],
        predicted_concepts=[]
    )
    with pytest.raises(NotFittedError):
        qp.predict(data)


def test_predict(qp, train_data):
    p_data = PredictData(
        docs=train_data.docs,
        predicted_concepts=train_data.predicted_concepts
    )

    qp.train(train_data)

    # Because of how our input data is designed,
    # we can make following assertion
    assert np.array_equal(qp.predict(p_data), [1] * 5)
