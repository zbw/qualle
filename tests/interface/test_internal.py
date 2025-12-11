#  Copyright 2021-2025 ZBW – Leibniz Information Centre for Economics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pathlib import Path

import pytest
from rdflib import URIRef
from sklearn.ensemble import GradientBoostingRegressor

import qualle.interface.internal as internal
import tests.interface.common as c
from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrationFeatures,
    SimpleLabelCalibrator,
)
from qualle.features.label_calibration.thesauri_label_calibration import (
    ThesauriLabelCalibrationFeatures,
    ThesauriLabelCalibrator,
)
from qualle.features.text import TextFeatures
from qualle.interface.config import (
    EvalSettings,
    FeaturesEnum,
    PredictSettings,
    RegressorSettings,
    SubthesauriLabelCalibrationSettings,
    TrainSettings,
)
from qualle.interface.internal import train
from qualle.models import PredictData, TrainData

TRAINER_CLS_FULL_PATH = "qualle.interface.internal.Trainer"
URI_PREFIX = "http://uri.tld/"


@pytest.fixture(autouse=True)
def mock_io(mocker):
    mocker.patch("qualle.interface.internal.dump")
    mocker.patch("qualle.interface.internal.load")


@pytest.fixture
def train_data():
    p = PredictData(
        docs=[f"Title{i}" for i in range(20)],
        predicted_labels=[[f"concept{i}"] for i in range(20)],
        scores=[[i / 20] for i in range(20)],
    )
    return TrainData(predict_data=p, true_labels=[[f"concept{i}"] for i in range(20)])


@pytest.fixture
def tsv_data_path(tmp_path):
    tsv_file = tmp_path / "doc.tsv"
    content = "\n".join(
        f"Title{i}\tconcept{i}: {i / 20}\tconcept{i}" for i in range(20)
    )
    tsv_file.write_text(content)
    return tsv_file


@pytest.fixture
def annif_data_dir(tmp_path):
    for i in range(20):
        doc0 = tmp_path / f"doc{i}.txt"
        doc0.write_text(f"Title{i}")
        scores0 = tmp_path / f"doc{i}.annif"
        scores0.write_text(f"<{URI_PREFIX}concept{i}>\tlabel0\t{i / 20}\n")

        labels0 = tmp_path / f"doc{i}.tsv"
        labels0.write_text(f"<{URI_PREFIX}concept{i}>\tlabel1\n")
    return tmp_path


@pytest.fixture
def train_settings(tsv_data_path):
    return TrainSettings(
        train_data_path=tsv_data_path,
        output_path="/tmp/output",
        should_debug=False,
        features=[FeaturesEnum.TEXT],
        label_calibrator_regressor=RegressorSettings(
            regressor_class="sklearn.ensemble.GradientBoostingRegressor",
            params=dict(n_estimators=10, max_depth=8),
        ),
        quality_estimator_regressor=RegressorSettings(
            regressor_class="sklearn.ensemble.GradientBoostingRegressor",
            params=dict(n_estimators=10, max_depth=8),
        ),
    )


def test_train_trains_trainer(train_settings, mocker):
    m_trainer = mocker.Mock()
    m_trainer_cls = mocker.Mock(return_value=m_trainer)
    m_trainer.train = mocker.Mock(return_value="testmodel")
    mocker.patch(TRAINER_CLS_FULL_PATH, m_trainer_cls)
    train(train_settings)

    m_trainer.train.assert_called_once()
    internal.dump.assert_called_once_with("testmodel", Path("/tmp/output"))


def test_train_without_slc_creates_respective_trainer(
    train_settings, mocker, train_data
):
    mocker.patch(TRAINER_CLS_FULL_PATH)

    train(train_settings)

    internal.Trainer.assert_called_once()
    call_args = internal.Trainer.call_args[1]
    assert call_args.get("train_data") == train_data
    lc = call_args.get("label_calibrator")
    assert isinstance(lc, SimpleLabelCalibrator)
    assert (
        lc.regressor.__dict__
        == GradientBoostingRegressor(**dict(n_estimators=10, max_depth=8)).__dict__
    )
    qr = call_args.get("quality_regressor")
    assert isinstance(qr, GradientBoostingRegressor)
    assert (
        qr.__dict__
        == GradientBoostingRegressor(**dict(n_estimators=10, max_depth=8)).__dict__
    )
    assert list(map(lambda f: f.__class__, call_args.get("features"))) == [
        TextFeatures,
        SimpleLabelCalibrationFeatures,
    ]
    assert call_args.get("should_debug") is False


def test_train_with_slc_creates_respective_trainer(
    train_settings, mocker, train_data, thsys_file_path
):
    m_graph = mocker.Mock()
    m_graph_cls = mocker.Mock(return_value=m_graph)
    mocker.patch("qualle.interface.internal.Graph", m_graph_cls)
    m_thesaurus = mocker.Mock()
    m_thesaurus_cls = mocker.Mock(return_value=m_thesaurus)
    mocker.patch("qualle.interface.internal.Thesaurus", m_thesaurus_cls)
    m_lcfst = mocker.Mock()
    m_lcfst_cls = mocker.Mock(return_value=m_lcfst)
    mocker.patch(
        "qualle.interface.internal.LabelCountForSubthesauriTransformer", m_lcfst_cls
    )
    mocker.patch(TRAINER_CLS_FULL_PATH)

    train_settings.subthesauri_label_calibration = SubthesauriLabelCalibrationSettings(
        thesaurus_file=thsys_file_path,
        subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
        concept_type=c.DUMMY_CONCEPT_TYPE,
        concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
        subthesauri=[c.DUMMY_SUBTHESAURUS_A, c.DUMMY_SUBTHESAURUS_B],
        use_sparse_count_matrix=True,
    )

    train(train_settings)

    m_graph_cls.assert_called_once()
    m_graph.parse.assert_called_once_with(thsys_file_path)

    m_thesaurus_cls.assert_called_once_with(
        graph=m_graph,
        subthesaurus_type_uri=URIRef(c.DUMMY_SUBTHESAURUS_TYPE),
        concept_type_uri=URIRef(c.DUMMY_CONCEPT_TYPE),
        concept_uri_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
    )

    m_lcfst_cls.assert_called_once_with(use_sparse_count_matrix=True)
    m_lcfst.init.assert_called_once_with(
        thesaurus=m_thesaurus,
        subthesauri=[URIRef(c.DUMMY_SUBTHESAURUS_A), URIRef(c.DUMMY_SUBTHESAURUS_B)],
    )

    internal.Trainer.assert_called_once()

    call_args = internal.Trainer.call_args[1]
    assert call_args.get("train_data") == train_data
    lc = call_args.get("label_calibrator")
    assert isinstance(lc, ThesauriLabelCalibrator)
    assert lc.regressor_class == GradientBoostingRegressor
    assert lc.regressor_params == dict(n_estimators=10, max_depth=8)
    qr = call_args.get("quality_regressor")
    assert isinstance(qr, GradientBoostingRegressor)
    assert (
        qr.__dict__
        == GradientBoostingRegressor(**dict(n_estimators=10, max_depth=8)).__dict__
    )
    assert list(map(lambda f: f.__class__, call_args.get("features"))) == [
        TextFeatures,
        ThesauriLabelCalibrationFeatures,
    ]
    assert call_args.get("should_debug") is False


def test_train_with_slc_uses_all_subthesauri_if_no_subthesauri_passed(
    train_settings, mocker, thsys_file_path
):
    m_graph = mocker.Mock()
    m_graph_cls = mocker.Mock(return_value=m_graph)
    mocker.patch("qualle.interface.internal.Graph", m_graph_cls)
    m_thesaurus = mocker.Mock()
    m_thesaurus.get_all_subthesauri.return_value = [URIRef(c.DUMMY_SUBTHESAURUS_B)]
    m_thesaurus_cls = mocker.Mock(return_value=m_thesaurus)
    mocker.patch("qualle.interface.internal.Thesaurus", m_thesaurus_cls)
    m_lcfst = mocker.Mock()
    m_lcfst_cls = mocker.Mock(return_value=m_lcfst)
    mocker.patch(
        "qualle.interface.internal.LabelCountForSubthesauriTransformer", m_lcfst_cls
    )
    mocker.patch(TRAINER_CLS_FULL_PATH)

    train_settings.subthesauri_label_calibration = SubthesauriLabelCalibrationSettings(
        thesaurus_file=thsys_file_path,
        subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
        concept_type=c.DUMMY_CONCEPT_TYPE,
        concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
        subthesauri=[],
        use_sparse_count_matrix=True,
    )

    train(train_settings)

    m_lcfst.init.assert_called_once_with(
        thesaurus=m_thesaurus,
        subthesauri=[URIRef(c.DUMMY_SUBTHESAURUS_B)],
    )


def test_evaluate(mocker, tsv_data_path, train_data, mdl_path):
    m_eval = mocker.Mock()
    m_eval.evaluate.return_value = {}
    m_eval_cls = mocker.Mock(return_value=m_eval)
    mocker.patch("qualle.interface.internal.Evaluator", m_eval_cls)
    internal.load.return_value = "testmodel"

    settings = EvalSettings(test_data_path=tsv_data_path, mdl_file=mdl_path)
    internal.evaluate(settings)

    m_eval_cls.assert_called_once_with(train_data, "testmodel")
    m_eval.evaluate.assert_called_once()


def test_load_train_input_from_annif(annif_data_dir, train_data):
    actual_train_data = internal._load_train_input(annif_data_dir)
    actual_train_data_tpls = zip(
        actual_train_data.predict_data.docs,
        actual_train_data.predict_data.predicted_labels,
        actual_train_data.predict_data.scores,
        actual_train_data.true_labels,
    )
    expected_train_data_tpls = zip(
        train_data.predict_data.docs,
        train_data.predict_data.predicted_labels,
        train_data.predict_data.scores,
        train_data.true_labels,
    )
    assert sorted(actual_train_data_tpls, key=lambda t: t[0]) == sorted(
        expected_train_data_tpls, key=lambda t: t[0]
    )


def test_load_train_input_from_tsv(tsv_data_path, train_data):
    assert internal._load_train_input(tsv_data_path) == train_data


def test_predict_stores_scores_from_model(tsv_data_path, tmp_path, mdl_path):
    output_path = tmp_path / "qualle.txt"
    settings = PredictSettings(
        predict_data_path=tsv_data_path, mdl_file=mdl_path, output_path=output_path
    )
    mock_model = internal.load.return_value
    mock_model.predict.side_effect = lambda p_data: map(lambda s: s[0], p_data.scores)

    internal.predict(settings)

    assert output_path.read_text().rstrip("\n") == "\n".join(
        [str(x / 20) for x in range(20)]
    )


def test_predict_with_annif_data_stores_scores_from_model(
    annif_data_dir, tmp_path, mdl_path
):
    settings = PredictSettings(
        predict_data_path=annif_data_dir,
        mdl_file=mdl_path,
    )
    mock_model = internal.load.return_value
    mock_model.predict.side_effect = lambda p_data: map(lambda s: s[0], p_data.scores)

    internal.predict(settings)

    for i in range(20):
        fp = annif_data_dir / f"doc{i}.qualle"
        assert fp.exists()
        assert fp.read_text() == str(i / 20), f"fail for {fp}"
