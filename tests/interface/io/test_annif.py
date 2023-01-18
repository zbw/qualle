#  Copyright 2021-2023 ZBW  – Leibniz Information Centre for Economics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import pytest
from pydantic import ValidationError

from qualle.interface.io.annif import AnnifHandler
from qualle.models import TrainData

DOC_0_CONTENT = 'title0\ncontent0'
DOC_1_CONTENT = 'title1\ncontent1'


_URI_PREFIX = 'http://uri.tld/'


@pytest.fixture
def data_without_true_labels(tmp_path):
    doc0 = tmp_path / 'doc0.txt'
    doc0.write_text(DOC_0_CONTENT)
    doc1 = tmp_path / 'doc1.txt'
    doc1.write_text(DOC_1_CONTENT)
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
def data(data_without_true_labels):
    labels0 = data_without_true_labels / 'doc0.tsv'
    labels0.write_text(
        f'<{_URI_PREFIX}concept1>\tlabel1\n'
        f'<{_URI_PREFIX}concept3>\tlabel3')
    labels1 = data_without_true_labels / 'doc1.tsv'
    labels1.write_text(f'{_URI_PREFIX}concept3>\tlabel3')
    return data_without_true_labels


@pytest.fixture
def data_with_empty_labels(tmp_path):
    doc0 = tmp_path / 'doc0.txt'
    doc0.write_text(DOC_0_CONTENT)
    doc1 = tmp_path / 'doc1.txt'
    doc1.write_text(DOC_1_CONTENT)
    scores0 = tmp_path / 'doc0.annif'
    scores0.write_text('')
    scores1 = tmp_path / 'doc1.annif'
    scores1.write_text(
        f'<{_URI_PREFIX}concept2>\tlabel2\t0\n'
        f'<{_URI_PREFIX}concept3>\tlabel3\t0.5')

    labels0 = tmp_path / 'doc0.tsv'
    labels0.write_text(
        f'<{_URI_PREFIX}concept1>\tlabel1\n'
        f'<{_URI_PREFIX}concept3>\tlabel3')
    labels1 = tmp_path / 'doc1.tsv'
    labels1.write_text('')
    return tmp_path


def test_load_train_input(data):
    handler = AnnifHandler(dir=data)
    parsed_input = handler.load_train_input()
    assert isinstance(parsed_input, TrainData)
    parsed_input_tpls = zip(
        parsed_input.predict_data.docs,
        parsed_input.predict_data.predicted_labels,
        parsed_input.predict_data.scores,
        parsed_input.true_labels
    )
    assert sorted(parsed_input_tpls, key=lambda t: t[0]) == [
        (DOC_0_CONTENT, ['concept0', 'concept1'], [1, .5],
         ['concept1', 'concept3']),
        (DOC_1_CONTENT, ['concept2', 'concept3'], [0., .5],
         ['concept3']),
    ]


def test_load_train_input_without_true_labels_raises_exc(
        data_without_true_labels):
    handler = AnnifHandler(dir=data_without_true_labels)
    with pytest.raises(ValidationError):
        handler.load_train_input()


def test_load_train_input_with_empty_labels_return_empty_list(
        data_with_empty_labels
):
    handler = AnnifHandler(dir=data_with_empty_labels)
    parsed_input = handler.load_train_input()
    assert isinstance(parsed_input, TrainData)
    parsed_input_tpls = zip(
        parsed_input.predict_data.docs,
        parsed_input.predict_data.predicted_labels,
        parsed_input.predict_data.scores,
        parsed_input.true_labels
    )
    assert sorted(parsed_input_tpls, key=lambda t: t[0]) == [
        (DOC_0_CONTENT, [], [],
         ['concept1', 'concept3']),
        (DOC_1_CONTENT, ['concept2', 'concept3'], [0, .5],
         []),
    ]


def test_extract_concept_id():
    assert 'concept_0' == AnnifHandler._extract_concept_id_from_annif_label(
        f'<{_URI_PREFIX}concept_0>'
    )
