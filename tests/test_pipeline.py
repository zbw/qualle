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

from qualle.models import TrainInput
from qualle.pipeline import recall, QualityEstimationPipeline


def test_train(mocker):
    concepts = [['c'] for _ in range(5)]
    data = TrainInput(
        docs=[f'd{i}' for i in range(5)],
        true_concepts=concepts,
        predicted_concepts=concepts
    )
    qp = QualityEstimationPipeline()
    spy = mocker.spy(qp._rp, 'fit')

    qp.train(data)

    actual_rp_input = spy.call_args[0][0]
    actual_true_recall = spy.call_args[0][1]

    # Because of how our input data is designed,
    # we can make following assertions
    only_ones = [1] * 5
    assert np.array_equal(actual_rp_input.no_of_pred_labels, only_ones)
    assert np.array_equal(actual_rp_input.label_calibration, only_ones)
    assert actual_true_recall == only_ones


def test_recall():
    true_concepts = [['x0', 'x2'], ['x1'], ['x3']]
    pred_concepts = [['x0', 'x1'], ['x2'], ['x3']]

    assert recall(
        true_concepts=true_concepts, predicted_concepts=pred_concepts) == [
        0.5, 0, 1]
