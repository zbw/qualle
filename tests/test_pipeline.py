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

from qualle.models import TrainData, PredictData
from qualle.pipeline import QualityEstimationPipeline


@pytest.fixture
def qp(mocker):
    label_calibrator = mocker.Mock()
    recall_predictor = mocker.Mock()
    mocker.patch(
        'qualle.pipeline.cross_val_predict',
        mocker.Mock(return_value=[1] * 5)
    )
    return QualityEstimationPipeline(
        label_calibrator=label_calibrator,
        rp=recall_predictor
    )


@pytest.fixture
def train_data():
    concepts = [['c'] for _ in range(5)]
    return TrainData(
        docs=[f'd{i}' for i in range(5)],
        true_concepts=concepts,
        predicted_concepts=concepts
    )


def test_train(qp, train_data, mocker):

    qp.train(train_data)

    qp._label_calibrator.fit.assert_called()
    qp._rp.fit.assert_called_once()

    actual_lc_docs = qp._label_calibrator.fit.call_args[0][0]
    actual_lc_true_concepts = qp._label_calibrator.fit.call_args[0][1]

    assert actual_lc_docs == train_data.docs
    assert actual_lc_true_concepts == train_data.true_concepts

    actual_label_calibration_data = qp._rp.fit.call_args[0][0]
    actual_true_recall = qp._rp.fit.call_args[0][1]

    # Because of how our input data is designed,
    # we can make following assertions
    only_ones = [1] * 5
    assert (actual_label_calibration_data.predicted_no_of_concepts
            == only_ones)
    assert actual_label_calibration_data.predicted_concepts == [['c']] * 5
    assert actual_true_recall == only_ones


def test_predict(qp, train_data):
    qp._rp.predict.return_value = [1] * 5
    p_data = PredictData(
        docs=train_data.docs,
        predicted_concepts=train_data.predicted_concepts
    )

    qp.train(train_data)

    # Because of how our input data is designed,
    # we can make following assertion
    assert np.array_equal(qp.predict(p_data), [1] * 5)
