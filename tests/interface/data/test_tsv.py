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

from qualle.interface.data.tsv import (
    load_predict_train_input,
    load_predict_input,
    RowParseError,
    load_label_calibration_train_input,
)
from qualle.models import (
    PredictTrainData,
    PredictData,
    LabelCalibrationTrainData,
)

import pytest

DOC_TSV = "doc.tsv"
LC_TRAIN_TSV = "lc_train.tsv"


@pytest.fixture
def predict_train_tsv_with_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text(
        "title0\tconcept0:1,concept1:0.5\tconcept1,concept3\n"
        "title1\tconcept2:0,concept3:0.5\tconcept3"
    )
    return tsv_file


@pytest.fixture
def predict_train_tsv_with_empty_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text(
        "title0\tconcept0:1,concept1:0.5\t\ntitle1\t\tconcept2,concept3"
    )
    return tsv_file


@pytest.fixture
def predict_train_tsv_without_true_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text(
        "title0\tconcept0:1,concept1:0.5\ntitle1\tconcept2:0,concept3:0.5"
    )
    return tsv_file


@pytest.fixture
def predict_tsv_with_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text(
        "title0\tconcept0:1,concept1:0.5\ntitle1\tconcept2:0,concept3:0.5"
    )
    return tsv_file


@pytest.fixture
def predict_tsv_with_empty_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text("title0\tconcept0:1,concept1:0.5\ntitle1\t")
    return tsv_file


@pytest.fixture
def predict_tsv_without_predicted_labels(tmp_path):
    tsv_file = tmp_path / DOC_TSV
    tsv_file.write_text("title0\ntitle1")
    return tsv_file


@pytest.fixture
def lc_train_tsv(tmp_path):
    tsv_file = tmp_path / LC_TRAIN_TSV
    tsv_file.write_text("title0\tconcept2,concept1\ntitle1\tconcept3")
    return tsv_file


@pytest.fixture
def lc_train_tsv_with_empty_labels(tmp_path):
    tsv_file = tmp_path / LC_TRAIN_TSV
    tsv_file.write_text("title0\t\ntitle1\t")
    return tsv_file


@pytest.fixture
def lc_train_tsv_without_true_labels(tmp_path):
    tsv_file = tmp_path / LC_TRAIN_TSV
    tsv_file.write_text("title0\ntitle1")
    return tsv_file


def test_load_predict_train_input(predict_train_tsv_with_labels):
    assert load_predict_train_input(predict_train_tsv_with_labels) == PredictTrainData(
        predict_data=PredictData(
            docs=["title0", "title1"],
            predicted_labels=[["concept0", "concept1"], ["concept2", "concept3"]],
            scores=[[1, 0.5], [0, 0.5]],
        ),
        true_labels=[["concept1", "concept3"], ["concept3"]],
    )


def test_load_predict_train_input_from_tsv_empty_labels_returns_empty_list(
    predict_train_tsv_with_empty_labels,
):
    assert load_predict_train_input(
        predict_train_tsv_with_empty_labels
    ) == PredictTrainData(
        predict_data=PredictData(
            docs=["title0", "title1"],
            predicted_labels=[["concept0", "concept1"], []],
            scores=[[1, 0.5], []],
        ),
        true_labels=[[], ["concept2", "concept3"]],
    )


def test_load_predict_train_input_without_true_labels_raises_exc(
    predict_train_tsv_without_true_labels,
):
    with pytest.raises(RowParseError):
        load_predict_train_input(predict_train_tsv_without_true_labels)


def test_load_predict_input(predict_tsv_with_labels):
    assert load_predict_input(predict_tsv_with_labels) == PredictData(
        docs=["title0", "title1"],
        predicted_labels=[["concept0", "concept1"], ["concept2", "concept3"]],
        scores=[[1, 0.5], [0, 0.5]],
    )


def test_load_predict_input_empty_labels_returns_empty_list(
    predict_tsv_with_empty_labels,
):
    assert load_predict_input(predict_tsv_with_empty_labels) == PredictData(
        docs=["title0", "title1"],
        predicted_labels=[["concept0", "concept1"], []],
        scores=[[1, 0.5], []],
    )


def test_load_predict_input_without_predicted_labels_raises_exc(
    predict_tsv_without_predicted_labels,
):
    with pytest.raises(RowParseError):
        load_predict_train_input(predict_tsv_without_predicted_labels)


def test_load_lc_train_input(lc_train_tsv):
    assert load_label_calibration_train_input(
        lc_train_tsv
    ) == LabelCalibrationTrainData(
        docs=["title0", "title1"], true_labels=[["concept2", "concept1"], ["concept3"]]
    )


def test_load_lc_train_input_empty_labels_returns_empty_list(
    lc_train_tsv_with_empty_labels,
):
    assert load_label_calibration_train_input(
        lc_train_tsv_with_empty_labels
    ) == LabelCalibrationTrainData(docs=["title0", "title1"], true_labels=[[], []])


def test_load_lc_train_input_without_true_labels_raises_exc(
    lc_train_tsv_without_true_labels,
):
    with pytest.raises(RowParseError):
        load_label_calibration_train_input(lc_train_tsv_without_true_labels)
