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

from qualle.interface.data.annif import (
    AnnifPredictTrainData,
    AnnifPredictData,
    load_predict_train_input,
    load_predict_input,
    store_quality_estimations,
    AnnifLabelCalibrationTrainData,
    load_label_calibration_train_input,
    AnnifLoadError,
)

DOC_0_NAME = "doc0"
DOC_1_NAME = "doc1"
DOC_0_ANNIF = f"{DOC_0_NAME}.annif"
DOC_1_ANNIF = f"{DOC_1_NAME}.annif"
DOC_0_TSV = f"{DOC_0_NAME}.tsv"
DOC_1_TSV = f"{DOC_1_NAME}.tsv"
DOC_0_TXT = f"{DOC_0_NAME}.txt"
DOC_1_TXT = f"{DOC_1_NAME}.txt"

DOC_0_CONTENT = "title0\ncontent0"
DOC_1_CONTENT = "title1\ncontent1"

URI_PREFIX = "http://uri.tld/"


@pytest.fixture
def dir_without_true_labels(tmp_path):
    doc0 = tmp_path / DOC_0_TXT
    doc0.write_text(DOC_0_CONTENT)
    doc1 = tmp_path / DOC_1_TXT
    doc1.write_text(DOC_1_CONTENT)
    scores0 = tmp_path / DOC_0_ANNIF
    scores0.write_text(
        f"<{URI_PREFIX}concept0>\tlabel0\t1\n<{URI_PREFIX}concept1>\tlabel1\t0.5"
    )
    scores1 = tmp_path / DOC_1_ANNIF
    scores1.write_text(
        f"<{URI_PREFIX}concept2>\tlabel2\t0\n<{URI_PREFIX}concept3>\tlabel3\t0.5"
    )
    return tmp_path


@pytest.fixture
def dir_without_predicted_labels(tmp_path):
    doc0 = tmp_path / DOC_0_TXT
    doc0.write_text(DOC_0_CONTENT)
    doc1 = tmp_path / DOC_1_TXT
    doc1.write_text(DOC_1_CONTENT)
    labels0 = tmp_path / DOC_0_TSV
    labels0.write_text(
        f"<{URI_PREFIX}concept1>\tlabel1\n<{URI_PREFIX}concept3>\tlabel3"
    )
    labels1 = tmp_path / DOC_1_TSV
    labels1.write_text(f"{URI_PREFIX}concept3>\tlabel3")
    return tmp_path


@pytest.fixture
def data_dir(dir_without_true_labels):
    labels0 = dir_without_true_labels / DOC_0_TSV
    labels0.write_text(
        f"<{URI_PREFIX}concept1>\tlabel1\n<{URI_PREFIX}concept3>\tlabel3"
    )
    labels1 = dir_without_true_labels / DOC_1_TSV
    labels1.write_text(f"{URI_PREFIX}concept3>\tlabel3")
    return dir_without_true_labels


@pytest.fixture
def dir_with_empty_labels(tmp_path):
    doc0 = tmp_path / DOC_0_TXT
    doc0.write_text(DOC_0_CONTENT)
    doc1 = tmp_path / DOC_1_TXT
    doc1.write_text(DOC_1_CONTENT)
    scores0 = tmp_path / DOC_0_ANNIF
    scores0.write_text("")
    scores1 = tmp_path / DOC_1_ANNIF
    scores1.write_text(
        f"<{URI_PREFIX}concept2>\tlabel2\t0\n<{URI_PREFIX}concept3>\tlabel3\t0.5"
    )

    labels0 = tmp_path / DOC_0_TSV
    labels0.write_text(
        f"<{URI_PREFIX}concept1>\tlabel1\n<{URI_PREFIX}concept3>\tlabel3"
    )
    labels1 = tmp_path / DOC_1_TSV
    labels1.write_text("")
    return tmp_path


def test_load_predict_train_input(data_dir):
    parsed_input = load_predict_train_input(data_dir)
    assert isinstance(parsed_input, AnnifPredictTrainData)

    assert set(parsed_input.document_ids) == {DOC_0_NAME, DOC_1_NAME}

    train_data = parsed_input.predict_train_data
    train_data_tpls = zip(
        train_data.predict_data.docs,
        train_data.predict_data.predicted_labels,
        train_data.predict_data.scores,
        train_data.true_labels,
    )
    assert sorted(train_data_tpls, key=lambda t: t[0]) == [
        (DOC_0_CONTENT, ["concept0", "concept1"], [1, 0.5], ["concept1", "concept3"]),
        (DOC_1_CONTENT, ["concept2", "concept3"], [0.0, 0.5], ["concept3"]),
    ]


def test_load_predict_train_input_without_true_labels_raises_exc(
    dir_without_true_labels,
):
    with pytest.raises(AnnifLoadError):
        load_predict_train_input(dir_without_true_labels)


def test_load_predict_train_input_with_empty_labels_return_empty_list(
    dir_with_empty_labels,
):
    parsed_input = load_predict_train_input(dir_with_empty_labels)
    assert isinstance(parsed_input, AnnifPredictTrainData)
    train_data = parsed_input.predict_train_data
    train_data_tpls = zip(
        train_data.predict_data.docs,
        train_data.predict_data.predicted_labels,
        train_data.predict_data.scores,
        train_data.true_labels,
    )
    assert sorted(train_data_tpls, key=lambda t: t[0]) == [
        (DOC_0_CONTENT, [], [], ["concept1", "concept3"]),
        (DOC_1_CONTENT, ["concept2", "concept3"], [0, 0.5], []),
    ]


def test_load_predict_input(data_dir):
    parsed_input = load_predict_input(data_dir)
    assert isinstance(parsed_input, AnnifPredictData)

    assert set(parsed_input.document_ids) == {DOC_0_NAME, DOC_1_NAME}

    predict_data_tpls = zip(
        parsed_input.predict_data.docs,
        parsed_input.predict_data.predicted_labels,
        parsed_input.predict_data.scores,
    )
    assert sorted(predict_data_tpls, key=lambda t: t[0]) == [
        (
            DOC_0_CONTENT,
            ["concept0", "concept1"],
            [1, 0.5],
        ),
        (
            DOC_1_CONTENT,
            ["concept2", "concept3"],
            [0.0, 0.5],
        ),
    ]


def test_load_predict_input_with_empty_labels_return_empty_list(dir_with_empty_labels):
    parsed_input = load_predict_input(dir_with_empty_labels)
    assert isinstance(parsed_input, AnnifPredictData)

    predict_data_tpls = zip(
        parsed_input.predict_data.docs,
        parsed_input.predict_data.predicted_labels,
        parsed_input.predict_data.scores,
    )
    assert sorted(predict_data_tpls, key=lambda t: t[0]) == [
        (
            DOC_0_CONTENT,
            [],
            [],
        ),
        (
            DOC_1_CONTENT,
            ["concept2", "concept3"],
            [0, 0.5],
        ),
    ]


def test_load_predict_input_without_predicted_labels_raises_exc(
    dir_without_predicted_labels,
):
    with pytest.raises(AnnifLoadError):
        load_predict_input(dir_without_predicted_labels)


def test_load_lc_train_input(dir_without_predicted_labels):
    parsed_input = load_label_calibration_train_input(dir_without_predicted_labels)
    assert isinstance(parsed_input, AnnifLabelCalibrationTrainData)

    assert set(parsed_input.document_ids) == {DOC_0_NAME, DOC_1_NAME}

    lc_train_data_tpls = zip(
        parsed_input.label_calibration_train_data.docs,
        parsed_input.label_calibration_train_data.true_labels,
    )
    assert sorted(lc_train_data_tpls, key=lambda t: t[0]) == [
        (
            DOC_0_CONTENT,
            ["concept1", "concept3"],
        ),
        (
            DOC_1_CONTENT,
            ["concept3"],
        ),
    ]


def test_load_lc_train_input_without_true_labels_raises_exc(dir_without_true_labels):
    with pytest.raises(AnnifLoadError):
        load_label_calibration_train_input(dir_without_true_labels)


def test_load_lc_train_input_with_empty_true_labels_return_empty_list(
    dir_with_empty_labels,
):
    parsed_input = load_label_calibration_train_input(dir_with_empty_labels)
    assert isinstance(parsed_input, AnnifLabelCalibrationTrainData)

    lc_train_data_tpls = zip(
        parsed_input.label_calibration_train_data.docs,
        parsed_input.label_calibration_train_data.true_labels,
    )
    assert sorted(lc_train_data_tpls, key=lambda t: t[0]) == [
        (
            DOC_0_CONTENT,
            ["concept1", "concept3"],
        ),
        (
            DOC_1_CONTENT,
            [],
        ),
    ]


def test_store_quality_ests_writes_files(data_dir):
    predict_data = load_predict_input(data_dir)

    scores = [0.5, 1]

    store_quality_estimations(data_dir, zip(scores, predict_data.document_ids))

    for i, doc_id in enumerate(predict_data.document_ids):
        qualle_fp = data_dir / (doc_id + ".qualle")
        assert qualle_fp.exists()
        assert qualle_fp.read_text() == str(scores[i])
