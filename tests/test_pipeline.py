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
        recall_predictor=recall_predictor
    )


@pytest.fixture
def train_data():
    labels = [['c'] for _ in range(5)]
    return TrainData(
        predict_data=PredictData(
            docs=[f'd{i}' for i in range(5)],
            predicted_labels=labels
        ),
        true_labels=labels,
    )


def test_train(qp, train_data, mocker):

    qp.train(train_data)

    qp._label_calibrator.fit.assert_called()
    qp._recall_predictor.fit.assert_called_once()

    actual_lc_docs = qp._label_calibrator.fit.call_args[0][0]
    actual_lc_true_labels = qp._label_calibrator.fit.call_args[0][1]

    assert actual_lc_docs == train_data.predict_data.docs
    assert actual_lc_true_labels == train_data.true_labels

    actual_label_calibration_data = qp._recall_predictor.fit.call_args[0][0]
    actual_true_recall = qp._recall_predictor.fit.call_args[0][1]

    # Because of how our input data is designed,
    # we can make following assertions
    only_ones = [1] * 5
    assert (actual_label_calibration_data.predicted_no_of_labels
            == only_ones)
    assert actual_label_calibration_data.predicted_labels == [['c']] * 5
    assert actual_true_recall == only_ones


def test_predict(qp, train_data):
    qp._recall_predictor.predict.return_value = [1] * 5
    p_data = train_data.predict_data

    qp.train(train_data)

    # Because of how our input data is designed,
    # we can make following assertion
    assert np.array_equal(qp.predict(p_data), [1] * 5)


def test_debug_prints_time_if_activated(qp, caplog):
    qp._should_debug = True
    caplog.set_level(logging.DEBUG)
    with qp._debug('msg'):
        _ = 1 + 1
    assert "Ran msg in " in caplog.text
    assert 'seconds' in caplog.text


def test_debug_prints_nothing_if_not_activated(qp, caplog):
    caplog.set_level(logging.DEBUG)
    with qp._debug('msg'):
        _ = 1 + 1
    assert "" == caplog.text
