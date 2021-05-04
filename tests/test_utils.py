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

from qualle.models import TrainData, PredictData
from qualle.utils import recall, load_train_input, train_input_from_tsv, \
    train_input_from_annif, timeit, extract_concept_id_from_annif_label
import pytest


_URI_PREFIX = 'http://uri.tld/'


def test_recall():
    true_labels = [['x0', 'x2'], ['x1'], ['x3']]
    pred_labels = [['x0', 'x1'], ['x2'], ['x3']]

    assert recall(
        true_labels=true_labels, predicted_labels=pred_labels) == [
        0.5, 0, 1]


def test_recall_empty_true_labels_return_zero():
    assert recall(true_labels=[[]], predicted_labels=[['x']]) == [0]


def test_recall_empty_pred_labels_return_zero():
    assert recall(true_labels=[['x']], predicted_labels=[[]]) == [0]


def test_recall_empty_input():
    assert recall(true_labels=[], predicted_labels=[]) == []


def test_load_train_input_selects_annif(mocker, tmp_path):
    m_train_input_from_annif = mocker.Mock()
    mocker.patch(
        'qualle.utils.train_input_from_annif', m_train_input_from_annif
    )
    m_train_input_from_tsv = mocker.Mock()
    mocker.patch(
        'qualle.utils.train_input_from_tsv', m_train_input_from_tsv
    )
    load_train_input(str(tmp_path))

    m_train_input_from_tsv.assert_not_called()
    m_train_input_from_annif.assert_called_once_with(str(tmp_path))


def test_load_train_input_selects_tsv(mocker, tmp_path):
    m_train_input_from_annif = mocker.Mock()
    mocker.patch(
        'qualle.utils.train_input_from_annif', m_train_input_from_annif
    )
    m_train_input_from_tsv = mocker.Mock()
    mocker.patch(
        'qualle.utils.train_input_from_tsv', m_train_input_from_tsv
    )
    load_train_input(str(tmp_path / 'empty.tsv'))

    m_train_input_from_annif.assert_not_called()
    m_train_input_from_tsv.assert_called_once_with(str(tmp_path / 'empty.tsv'))


def test_train_input_from_tsv(mocker):
    m = mocker.mock_open(
        read_data='title0\tconcept0:1,concept1:0.5\tconcept1,concept3\n'
                  'title1\tconcept2:0,concept3:0.5\tconcept3'
    )
    mocker.patch('qualle.utils.open', m)

    assert train_input_from_tsv('dummypath') == TrainData(
        predict_data=PredictData(
            docs=['title0', 'title1'],
            predicted_labels=[
                ['concept0', 'concept1'], ['concept2', 'concept3']
            ],
            scores=[[1, .5], [0, .5]]
        ),
        true_labels=[['concept1', 'concept3'], ['concept3']]
    )


def test_train_input_from_tsv_empty_labels__returns_empty_list(mocker):
    m = mocker.mock_open(
        read_data='title0\t\tconcept1,concept3\n'
                  'title1\tconcept2:1,concept3:0.5\t'
    )
    mocker.patch('qualle.utils.open', m)

    assert train_input_from_tsv('dummypath') == TrainData(
        predict_data=PredictData(
            docs=['title0', 'title1'],
            predicted_labels=[
                [], ['concept2', 'concept3']
            ],
            scores=[
                [], [1, .5]
            ]
        ),
        true_labels=[['concept1', 'concept3'], []]
    )


@pytest.fixture
def annif_data_without_true_labels(tmp_path):
    doc0 = tmp_path / 'doc0.txt'
    doc0.write_text('title0\ncontent0')
    doc1 = tmp_path / 'doc1.txt'
    doc1.write_text('title1\ncontent1')
    scores0 = tmp_path / 'doc0.annif'
    scores0.write_text(
        f'<{_URI_PREFIX}concept0>\tlabel0\t1\n'
        f'<{_URI_PREFIX}concept1>\tlabel1\t0.5')
    scores1 = tmp_path / 'doc1.annif'
    scores1.write_text(
        f'<{_URI_PREFIX}concept2>\tlabel2\t0\n'
        f'<{_URI_PREFIX}concept3>\tlabel3\t0.5')
    return tmp_path


@pytest.fixture
def annif_data_with_labels(
        annif_data_without_true_labels):
    labels0 = annif_data_without_true_labels / 'doc0.tsv'
    labels0.write_text(
        f'<{_URI_PREFIX}concept1>\tlabel1\n'
        f'<{_URI_PREFIX}concept3>\tlabel3')
    labels1 = annif_data_without_true_labels / 'doc1.tsv'
    labels1.write_text(f'{_URI_PREFIX}concept3>\tlabel3')
    return annif_data_without_true_labels


def test_train_input_from_annif(
        annif_data_with_labels):
    assert train_input_from_annif(str(annif_data_with_labels)) == TrainData(
        predict_data=PredictData(
            docs=['title0\ncontent0', 'title1\ncontent1'],
            predicted_labels=[
                ['concept0', 'concept1'], ['concept2', 'concept3']
            ],
            scores=[[1, .5], [0, .5]]
        ),
        true_labels=[['concept1', 'concept3'], ['concept3']]
    )


def test_train_input_from_annif_without_labels_returns_empty_list(
        annif_data_without_true_labels):
    assert train_input_from_annif(
            str(annif_data_without_true_labels)) == TrainData(
        predict_data=PredictData(
            docs=['title0\ncontent0', 'title1\ncontent1'],
            predicted_labels=[
                ['concept0', 'concept1'], ['concept2', 'concept3']
            ],
            scores=[[1, .5], [0, .5]]
        ),
        true_labels=[]
    )


def test_extract_concept_id():
    assert 'concept_0' == extract_concept_id_from_annif_label(
        f'<{_URI_PREFIX}concept_0>'
    )


def test_timeit(mocker):
    m = mocker.Mock(side_effect=[1, 3])
    mocker.patch('qualle.utils.perf_counter', m)
    with timeit() as t:
        _ = 1 + 1
    assert t() == 2
