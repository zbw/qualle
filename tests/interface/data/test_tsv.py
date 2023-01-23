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
from pydantic import ValidationError

from qualle.interface.data.tsv import load_train_input, load_predict_input
from qualle.models import TrainData, PredictData

import pytest

DOC_TSV = "doc.tsv"


@pytest.fixture
def data_with_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text(
        "title0\tconcept0:1,concept1:0.5\tconcept1,concept3\n"
        "title1\tconcept2:0,concept3:0.5\tconcept3"
    )
    return tsv_file


@pytest.fixture
def data_with_empty_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text(
        "title0\tconcept0:1,concept1:0.5\t\n" "title1\t\tconcept2,concept3"
    )
    return tsv_file


@pytest.fixture
def data_without_true_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text(
        "title0\tconcept0:1,concept1:0.5\n" "title1\tconcept2:0,concept3:0.5"
    )
    return tsv_file


def test_load_train_input(data_with_labels):
    assert load_train_input(data_with_labels) == TrainData(
        predict_data=PredictData(
            docs=["title0", "title1"],
            predicted_labels=[["concept0", "concept1"], ["concept2", "concept3"]],
            scores=[[1, 0.5], [0, 0.5]],
        ),
        true_labels=[["concept1", "concept3"], ["concept3"]],
    )


def test_load_train_input_from_tsv_empty_labels_returns_empty_list(
    data_with_empty_labels,
):
    assert load_train_input(data_with_empty_labels) == TrainData(
        predict_data=PredictData(
            docs=["title0", "title1"],
            predicted_labels=[["concept0", "concept1"], []],
            scores=[[1, 0.5], []],
        ),
        true_labels=[[], ["concept2", "concept3"]],
    )


def test_load_train_input_without_true_labels_raises_exc(data_without_true_labels):
    with pytest.raises(ValidationError):
        load_train_input(data_without_true_labels)


def test_load_predict_data(data_with_labels):
    assert load_predict_input(data_with_labels) == PredictData(
        docs=["title0", "title1"],
        predicted_labels=[["concept0", "concept1"], ["concept2", "concept3"]],
        scores=[[1, 0.5], [0, 0.5]],
    )


def test_load_predict_input_from_tsv_empty_labels_returns_empty_list(
    data_with_empty_labels,
):
    assert load_predict_input(data_with_empty_labels) == PredictData(
        docs=["title0", "title1"],
        predicted_labels=[["concept0", "concept1"], []],
        scores=[[1, 0.5], []],
    )
