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
import argparse
import pytest
import sys
import logging
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
from pathlib import Path

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


def test_cli_entrypoint_with_eval_parser(mocker, monkeypatch, tmp_path, mdl_path):
    test_data_path = tmp_path / "testdata"
    test_data_path.mkdir()
    mock_eval_func = mocker.patch("qualle.interface.cli.handle_eval")

    # config_logging() method needs to be mocked otherwise it will be called and
    # disturb the global settings for the logger used in qualle. This results in
    # a failed unit test namely, test_debug_prints_time_if_activated() inside tests/test_pipeline.py
    _ = mocker.patch("qualle.interface.cli.config_logging", return_value="foo")

    monkeypatch.setattr(
        sys, "argv", ["cli.py", "eval", str(test_data_path), str(mdl_path)]
    )
    cli.cli_entrypoint()
    args_passed = mock_eval_func.call_args[0][0]
    assert Path(args_passed.test_data_path) == test_data_path
    assert Path(args_passed.model) == mdl_path


def test_cli_entrypoint_with_rest_parser(mocker, monkeypatch, mdl_path):
    mock_rest_func = mocker.patch("qualle.interface.cli.handle_rest")

    # config_logging() method needs to be mocked otherwise it will be called and
    # disturb the global settings for the logger used in qualle. Without mocking this method
    # a unit test fails, namely test_debug_prints_time_if_activated() inside tests/test_pipeline.py
    _ = mocker.patch("qualle.interface.cli.config_logging", return_value="foo")

    monkeypatch.setattr(
        sys, "argv", ["cli.py", "rest", str(mdl_path), "--host=x", "--port=9000"]
    )
    cli.cli_entrypoint()
    args_passed = mock_rest_func.call_args[0][0]
    assert Path(args_passed.model) == mdl_path
    assert args_passed.host == ["x"]
    assert args_passed.port == [9000]


def test_cli_entrypoint_with_predict_parser(
    mocker, monkeypatch, tmp_path, tsv_file_path, mdl_path
):
    output_path = tmp_path / "output.txt"
    mock_predict_func = mocker.patch("qualle.interface.cli.handle_predict")

    # config_logging() method needs to be mocked otherwise it will be called and
    # disturb the global settings for the logger used in qualle. Without mocking this method
    # a unit test fails, namely test_debug_prints_time_if_activated() inside tests/test_pipeline.py
    _ = mocker.patch("qualle.interface.cli.config_logging", return_value="foo")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "predict",
            str(tsv_file_path),
            str(mdl_path),
            "--output=" + str(output_path),
        ],
    )
    cli.cli_entrypoint()
    args_passed = mock_predict_func.call_args[0][0]
    assert Path(args_passed.predict_data_path) == tsv_file_path
    assert Path(args_passed.model) == mdl_path
    assert args_passed.output == [output_path]


def test_cli_entrypoint_for_train_parser_without_slc(
    mocker, monkeypatch, train_args_dict
):
    mock_train_func = mocker.patch("qualle.interface.cli.handle_train")

    # config_logging() method needs to be mocked otherwise it will be called and
    # disturb the global settings for the logger used in qualle. Without mocking this method
    # a unit test fails, namely test_debug_prints_time_if_activated() inside tests/test_pipeline.py
    _ = mocker.patch("qualle.interface.cli.config_logging", return_value="foo")

    train_data_path = train_args_dict["train_data_path"]
    output = train_args_dict["output"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "train",
            str(train_data_path),
            str(output),
            "--label-calibrator-regressor="
            + train_args_dict["label_calibrator_regressor"][0],
            "--quality-estimator-regressor="
            + train_args_dict["quality_estimator_regressor"][0],
        ],
    )

    cli.cli_entrypoint()
    args_passed = mock_train_func.call_args[0][0]
    assert Path(args_passed.train_data_path) == train_args_dict["train_data_path"]
    assert Path(args_passed.output) == Path(train_args_dict["output"])
    assert (
        args_passed.label_calibrator_regressor
        == train_args_dict["label_calibrator_regressor"]
    )
    assert (
        args_passed.quality_estimator_regressor
        == train_args_dict["quality_estimator_regressor"]
    )


def test_cli_entrypoint_for_train_parser_with_slc(
    mocker, monkeypatch, train_args_dict_with_slc
):
    mock_train_func = mocker.patch("qualle.interface.cli.handle_train")

    # config_logging() method needs to be mocked otherwise it will be called and
    # disturb the global settings for the logger used in qualle. Without mocking this method
    # a unit test fails, namely test_debug_prints_time_if_activated() inside tests/test_pipeline.py
    _ = mocker.patch("qualle.interface.cli.config_logging", return_value="foo")

    train_data_path = train_args_dict_with_slc["train_data_path"]
    output = train_args_dict_with_slc["output"]
    thsys_data_path = train_args_dict_with_slc["thsys"][0]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "train",
            str(train_data_path),
            str(output),
            "--label-calibrator-regressor="
            + train_args_dict_with_slc["label_calibrator_regressor"][0],
            "--quality-estimator-regressor="
            + train_args_dict_with_slc["quality_estimator_regressor"][0],
            "--slc",
            "--thsys=" + str(thsys_data_path),
            "--s-type=" + train_args_dict_with_slc["s_type"][0],
            "--c-type=" + train_args_dict_with_slc["c_type"][0],
            "--c-uri-prefix=" + train_args_dict_with_slc["c_uri_prefix"][0],
        ],
    )

    cli.cli_entrypoint()
    args_passed = mock_train_func.call_args[0][0]
    assert (
        Path(args_passed.train_data_path) == train_args_dict_with_slc["train_data_path"]
    )
    assert Path(args_passed.output) == Path(train_args_dict_with_slc["output"])
    assert (
        args_passed.label_calibrator_regressor
        == train_args_dict_with_slc["label_calibrator_regressor"]
    )
    assert (
        args_passed.quality_estimator_regressor
        == train_args_dict_with_slc["quality_estimator_regressor"]
    )
    assert args_passed.slc
    assert Path(args_passed.thsys[0]) == train_args_dict_with_slc["thsys"][0]
    assert args_passed.s_type == train_args_dict_with_slc["s_type"]
    assert args_passed.c_type == train_args_dict_with_slc["c_type"]
    assert args_passed.c_uri_prefix == train_args_dict_with_slc["c_uri_prefix"]
    assert not args_passed.use_sparse_count_matrix


def test_add_eval_parser(mocker, tmp_path, mdl_path):
    test_data_path = tmp_path / "testdata"
    test_data_path.mkdir()
    mock_eval_func = mocker.patch("qualle.interface.cli.handle_eval")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Subcommands", required=True, dest="command"
    )
    cli.add_eval_parser(subparsers)
    args = parser.parse_args(["eval", str(test_data_path), str(mdl_path)])

    assert args.command == "eval"
    assert Path(args.test_data_path) == test_data_path
    assert Path(args.model) == mdl_path
    assert args.func == mock_eval_func

    args.func(args)
    mock_eval_func.assert_called_once_with(args)


def test_add_train_parser(mocker, train_args_dict):
    train_data_path = train_args_dict["train_data_path"]
    output = train_args_dict["output"]
    mock_train_func = mocker.patch("qualle.interface.cli.handle_train")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Subcommands", required=True, dest="command"
    )
    cli.add_train_parser(subparsers)
    args = parser.parse_args(
        [
            "train",
            str(train_data_path),
            str(output),
            "--label-calibrator-regressor="
            + train_args_dict["label_calibrator_regressor"][0],
            "--quality-estimator-regressor="
            + train_args_dict["quality_estimator_regressor"][0],
        ]
    )
    assert args.command == "train"
    assert Path(args.train_data_path) == train_args_dict["train_data_path"]
    assert Path(args.output) == Path(train_args_dict["output"])
    assert (
        args.label_calibrator_regressor == train_args_dict["label_calibrator_regressor"]
    )
    assert (
        args.quality_estimator_regressor
        == train_args_dict["quality_estimator_regressor"]
    )
    assert args.func == mock_train_func

    args.func(args)
    mock_train_func.assert_called_once_with(args)


def test_add_slc_group(train_args_dict_with_slc):
    thsys_data_path = train_args_dict_with_slc["thsys"][0]
    parser = argparse.ArgumentParser()
    cli.add_slc_group(parser)
    args = parser.parse_args(
        [
            "--slc",
            "--thsys=" + str(thsys_data_path),
            "--s-type=" + train_args_dict_with_slc["s_type"][0],
            "--c-type=" + train_args_dict_with_slc["c_type"][0],
            "--c-uri-prefix=" + train_args_dict_with_slc["c_uri_prefix"][0],
        ]
    )
    assert args.slc
    assert Path(args.thsys[0]) == train_args_dict_with_slc["thsys"][0]
    assert args.s_type == train_args_dict_with_slc["s_type"]
    assert args.c_type == train_args_dict_with_slc["c_type"]
    assert args.c_uri_prefix == train_args_dict_with_slc["c_uri_prefix"]
    assert not args.use_sparse_count_matrix

    group_titles = [group.title for group in parser._action_groups]
    assert "subthesauri-label-calibration" in group_titles


def test_add_rest_parser(mocker, mdl_path):
    mock_rest_func = mocker.patch("qualle.interface.cli.handle_rest")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Subcommands", required=True, dest="command"
    )
    cli.add_rest_parser(subparsers)
    args = parser.parse_args(["rest", str(mdl_path), "--host=x", "--port=9000"])

    assert args.command == "rest"
    assert Path(args.model) == mdl_path
    assert args.host == ["x"]
    assert args.port == [9000]
    assert args.func == mock_rest_func

    args.func(args)
    mock_rest_func.assert_called_once_with(args)


def test_add_predict_parser(mocker, tmp_path, tsv_file_path, mdl_path):
    output_path = tmp_path / "output.txt"
    mock_predict_func = mocker.patch("qualle.interface.cli.handle_predict")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Subcommands", required=True, dest="command"
    )
    cli.add_predict_parser(subparsers)
    args = parser.parse_args(
        ["predict", str(tsv_file_path), str(mdl_path), "--output=" + str(output_path)]
    )

    assert args.command == "predict"
    assert Path(args.predict_data_path) == tsv_file_path
    assert Path(args.model) == mdl_path
    assert args.output == [output_path]
    assert args.func == mock_predict_func

    args.func(args)
    mock_predict_func.assert_called_once_with(args)


def test_config_logging_with_file(mocker, tmp_path):
    config_file = tmp_path / "logging.conf"
    config_file.touch()

    dummy_logger = logging.getLogger("foo")
    dummy_logger.addHandler(logging.NullHandler())

    mocker.patch("qualle.interface.cli.get_logger", return_value=dummy_logger)
    mock_config_file = mocker.patch("qualle.interface.cli.logging.config.fileConfig")
    cli.config_logging(config_file)
    mock_config_file.assert_called_once_with(config_file)


def test_config_logging_with_debug(mocker):
    dummy_logger = logging.getLogger("bar")
    dummy_logger.addHandler(logging.NullHandler())

    mocker.patch("qualle.interface.cli.get_logger", return_value=dummy_logger)

    cli.config_logging(debug=True)
    handlers = dummy_logger.handlers
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            assert h.level == logging.DEBUG

    assert dummy_logger.isEnabledFor(logging.DEBUG)


def test_config_logging_without_debug(mocker):
    dummy_logger = logging.getLogger("baz")
    dummy_logger.addHandler(logging.NullHandler())

    mocker.patch("qualle.interface.cli.get_logger", return_value=dummy_logger)

    cli.config_logging()
    handlers = dummy_logger.handlers
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            assert h.level == logging.INFO
    assert dummy_logger.isEnabledFor(logging.INFO)
