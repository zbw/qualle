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

from qualle.models import TrainData
from qualle.utils import recall, train_input_from_tsv


def test_recall():
    true_concepts = [['x0', 'x2'], ['x1'], ['x3']]
    pred_concepts = [['x0', 'x1'], ['x2'], ['x3']]

    assert recall(
        true_concepts=true_concepts, predicted_concepts=pred_concepts) == [
        0.5, 0, 1]


def test_recall_empty_true_concepts_return_zero():
    assert recall(true_concepts=[[]], predicted_concepts=[['x']]) == [0]


def test_recall_empty_pred_concepts_return_zero():
    assert recall(true_concepts=[['x']], predicted_concepts=[[]]) == [0]


def test_recall_empty_input():
    assert recall(true_concepts=[], predicted_concepts=[]) == []


def test_train_input_from_tsv(mocker):
    m = mocker.mock_open(
        read_data='title0\tconcept0:1,concept1:0.5\tconcept1,concept3\n'
                  'title1\tconcept2:1,concept3:0.5\tconcept3'
    )
    mocker.patch('qualle.utils.open', m)

    assert train_input_from_tsv('dummypath') == TrainData(
        docs=['title0', 'title1'],
        predicted_concepts=[
            ['concept0', 'concept1'], ['concept2', 'concept3']
        ],
        true_concepts=[['concept1', 'concept3'], ['concept3']]
    )
