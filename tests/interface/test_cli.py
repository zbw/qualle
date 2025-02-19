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
from argparse import Namespace

import pytest

import qualle.interface.cli as cli
from qualle.interface.config import (
    FeaturesEnum,
    RegressorSettings,
    SubthesauriLabelCalibrationSettings,
    TrainSettings,
    EvalSettings,
    RESTSettings,
    PredictSettings,
)
from qualle.interface.cli import CliValidationError, handle_train, handle_eval

import tests.interface.common as c


@pytest.fixture
def train_args_dict(tmp_path):
    train_data_path = tmp_path / "train"
    train_data_path.mkdir()
    return dict(
        train_data_path=train_data_path,
        output="/tmp/output",
        slc=False,
        should_debug=False,
        features=[],
        label_calibrator_regressor=[
            '{"class": "sklearn.ensemble.GradientBoostingRegressor",'
            '"min_samples_leaf": 30, "max_depth": 5, "n_estimators": 10}'
        ],
        quality_estimator_regressor=[
            '{"class": "sklearn.ensemble.ExtraTreesRegressor"}'
        ],
    )


@pytest.fixture
def train_args_dict_with_slc(train_args_dict, thsys_file_path):
    train_args_dict["slc"] = True
    train_args_dict["thsys"] = [thsys_file_path]
    train_args_dict["s_type"] = [c.DUMMY_SUBTHESAURUS_TYPE]
    train_args_dict["c_uri_prefix"] = [c.DUMMY_CONCEPT_TYPE_PREFIX]
    train_args_dict["c_type"] = [c.DUMMY_CONCEPT_TYPE]
    train_args_dict["subthesauri"] = []
    train_args_dict["use_sparse_count_matrix"] = False

    return train_args_dict


@pytest.fixture(autouse=True)
def mock_internal_interface(mocker):
    mocker.patch("qualle.interface.cli.train")
    mocker.patch("qualle.interface.cli.evaluate")
    mocker.patch("qualle.interface.cli.predict")


@pytest.fixture
def tsv_file_path(tmp_path):
    fp = tmp_path / "doc.tsv"
    fp.write_text("t\tc:0\tc")
    return fp


def test_handle_train_slc_without_all_required_args_raises_exc(train_args_dict):
    train_args_dict["slc"] = True
    train_args_dict["thsys"] = train_args_dict["s_type"] = train_args_dict[
        "c_uri_prefix"
    ] = None
    train_args_dict["c_type"] = "http://test"

    with pytest.raises(CliValidationError):
        handle_train(Namespace(**train_args_dict))


def test_handle_train_slc_with_subthesauri(train_args_dict_with_slc, thsys_file_path):
    train_args_dict_with_slc["subthesauri"] = [
        ",".join((c.DUMMY_SUBTHESAURUS_A, c.DUMMY_SUBTHESAURUS_B))
    ]

    handle_train(Namespace(**train_args_dict_with_slc))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert (
        actual_settings.subthesauri_label_calibration
        == SubthesauriLabelCalibrationSettings(
            thesaurus_file=thsys_file_path,
            subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
            concept_type=c.DUMMY_CONCEPT_TYPE,
            concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
            subthesauri=[c.DUMMY_SUBTHESAURUS_A, c.DUMMY_SUBTHESAURUS_B],
        )
    )


def test_handle_train_slc_without_subthesauri(
    train_args_dict_with_slc, thsys_file_path
):
    handle_train(Namespace(**train_args_dict_with_slc))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert (
        actual_settings.subthesauri_label_calibration
        == SubthesauriLabelCalibrationSettings(
            thesaurus_file=thsys_file_path,
            subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
            concept_type=c.DUMMY_CONCEPT_TYPE,
            concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
            subthesauri=[],
        )
    )


def test_handle_train_slc_with_sparse_count_matrix(
    train_args_dict_with_slc, thsys_file_path
):
    train_args_dict_with_slc["use_sparse_count_matrix"] = True

    handle_train(Namespace(**train_args_dict_with_slc))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert (
        actual_settings.subthesauri_label_calibration
        == SubthesauriLabelCalibrationSettings(
            thesaurus_file=thsys_file_path,
            subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
            concept_type=c.DUMMY_CONCEPT_TYPE,
            concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
            subthesauri=[],
            use_sparse_count_matrix=True,
        )
    )


def test_handle_train_without_slc(train_args_dict):
    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.subthesauri_label_calibration is None


def test_handle_train_all_features(train_args_dict):
    train_args_dict["features"] = ["all"]

    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.features == [FeaturesEnum.CONFIDENCE, FeaturesEnum.TEXT]


def test_handle_train_confidence_features(train_args_dict):
    train_args_dict["features"] = ["confidence"]

    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.features == [FeaturesEnum.CONFIDENCE]


def test_handle_train_no_features(train_args_dict):
    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.features == []


def test_handle_train_creates_regressors(train_args_dict):
    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.label_calibrator_regressor == RegressorSettings(
        regressor_class="sklearn.ensemble.GradientBoostingRegressor",
        params=dict(min_samples_leaf=30, max_depth=5, n_estimators=10),
    )
    assert actual_settings.quality_estimator_regressor == RegressorSettings(
        regressor_class="sklearn.ensemble.ExtraTreesRegressor", params=dict()
    )


def test_handle_eval(tmp_path, mdl_path):
    test_data_path = tmp_path / "testdata"
    test_data_path.mkdir()
    handle_eval(Namespace(**dict(test_data_path=test_data_path, model=mdl_path)))
    cli.evaluate.assert_called_once()
    actual_settings = cli.evaluate.call_args[0][0]
    assert actual_settings == EvalSettings(
        test_data_path=test_data_path, mdl_file=mdl_path
    )


def test_handle_rest(mocker, mdl_path):
    m_run = mocker.Mock()
    mocker.patch("qualle.interface.cli.run", m_run)

    cli.handle_rest(Namespace(**dict(model=mdl_path, port=[9000], host=["x"])))

    m_run.assert_called_once_with(RESTSettings(mdl_file=mdl_path, host="x", port=9000))


def test_handle_predict_with_dir(tmp_path, mdl_path):
    predict_data_path = tmp_path / "predict"
    predict_data_path.mkdir()
    cli.handle_predict(
        Namespace(
            **dict(predict_data_path=predict_data_path, model=mdl_path, output=None)
        )
    )
    cli.predict.assert_called_once()
    actual_settings = cli.predict.call_args[0][0]
    assert actual_settings == PredictSettings(
        predict_data_path=predict_data_path, mdl_file=mdl_path
    )


def test_handle_predict_with_file(tsv_file_path, tmp_path, mdl_path):
    output_path = tmp_path / "output.txt"
    cli.handle_predict(
        Namespace(
            **dict(
                predict_data_path=tsv_file_path, model=mdl_path, output=[output_path]
            )
        )
    )
    cli.predict.assert_called_once()
    actual_settings = cli.predict.call_args[0][0]
    assert actual_settings == PredictSettings(
        predict_data_path=tsv_file_path, mdl_file=mdl_path, output_path=output_path
    )


def test_handle_predict_with_file_raises_exc_if_no_output_file(tsv_file_path, mdl_path):
    with pytest.raises(CliValidationError):
        cli.handle_predict(
            Namespace(
                **dict(predict_data_path=tsv_file_path, model=mdl_path, output=None)
            )
        )
