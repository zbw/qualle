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
from argparse import Namespace

import pytest

import qualle.interface.cli as cli
from qualle.interface.config import FeaturesEnum, RegressorSettings, \
    SubthesauriLabelCalibrationSettings, TrainSettings, EvalSettings, \
    RESTSettings
from qualle.interface.cli import CliValidationError, handle_train, handle_eval

import tests.interface.common as c

DUMMY_MODEL_PATH = '/tmp/model'


@pytest.fixture
def train_args_dict():
    return dict(
        train_data_path='/tmp/train',
        output='/tmp/output',
        slc=False,
        should_debug=False,
        features=[],
        label_calibrator_regressor=[
         '{"class": "sklearn.ensemble.GradientBoostingRegressor",'
         '"min_samples_leaf": 30, "max_depth": 5, "n_estimators": 10}'
        ],
        quality_estimator_regressor=[
            '{"class": "sklearn.ensemble.ExtraTreesRegressor"}'
        ]
    )


@pytest.fixture(autouse=True)
def mock_internal_interface(mocker):
    mocker.patch('qualle.interface.cli.train')
    mocker.patch('qualle.interface.cli.evaluate')


def test_handle_train_slc_without_all_required_args_raises_exc(
        train_args_dict
):
    train_args_dict['slc'] = True
    train_args_dict['thsys'] = train_args_dict['s_type'] = train_args_dict[
        'c_uri_prefix'] = None
    train_args_dict['c_type'] = 'http://test'

    with pytest.raises(CliValidationError):
        handle_train(Namespace(**train_args_dict))


def test_handle_train_slc_with_subthesauri(train_args_dict):
    train_args_dict['slc'] = True
    train_args_dict['thsys'] = [c.DUMMY_THESAURUS_FILE]
    train_args_dict['s_type'] = [c.DUMMY_SUBTHESAURUS_TYPE]
    train_args_dict['c_uri_prefix'] = [c.DUMMY_CONCEPT_TYPE_PREFIX]
    train_args_dict['c_type'] = [c.DUMMY_CONCEPT_TYPE]
    train_args_dict['subthesauri'] = [
        ','.join((c.DUMMY_SUBTHESAURUS_A, c.DUMMY_SUBTHESAURUS_B))]

    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.subthesauri_label_calibration ==\
           SubthesauriLabelCalibrationSettings(
                thesaurus_file=c.DUMMY_THESAURUS_FILE,
                subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
                concept_type=c.DUMMY_CONCEPT_TYPE,
                concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
                subthesauri=[c.DUMMY_SUBTHESAURUS_A, c.DUMMY_SUBTHESAURUS_B]
            )


def test_handle_train_slc_without_subthesauri(train_args_dict):
    train_args_dict['slc'] = True
    train_args_dict['thsys'] = [c.DUMMY_THESAURUS_FILE]
    train_args_dict['s_type'] = [c.DUMMY_SUBTHESAURUS_TYPE]
    train_args_dict['c_uri_prefix'] = [c.DUMMY_CONCEPT_TYPE_PREFIX]
    train_args_dict['c_type'] = [c.DUMMY_CONCEPT_TYPE]
    train_args_dict['subthesauri'] = []

    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.subthesauri_label_calibration ==\
           SubthesauriLabelCalibrationSettings(
                thesaurus_file=c.DUMMY_THESAURUS_FILE,
                subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
                concept_type=c.DUMMY_CONCEPT_TYPE,
                concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
                subthesauri=[]
            )


def test_handle_train_without_slc(train_args_dict):
    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.subthesauri_label_calibration is None


def test_handle_train_all_features(train_args_dict):
    train_args_dict['features'] = ['all']

    handle_train(Namespace(**train_args_dict))

    cli.train.assert_called_once()
    actual_settings = cli.train.call_args[0][0]

    assert isinstance(actual_settings, TrainSettings)
    assert actual_settings.features == [
        FeaturesEnum.CONFIDENCE, FeaturesEnum.TEXT
    ]


def test_handle_train_confidence_features(train_args_dict):
    train_args_dict['features'] = ['confidence']

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
        regressor_class='sklearn.ensemble.GradientBoostingRegressor',
        params=dict(min_samples_leaf=30, max_depth=5, n_estimators=10)
    )
    assert actual_settings.quality_estimator_regressor == RegressorSettings(
        regressor_class='sklearn.ensemble.ExtraTreesRegressor',
        params=dict()
    )


def test_handle_eval():
    handle_eval(
        Namespace(**dict(test_data_path='/tmp/test', model=DUMMY_MODEL_PATH)))
    cli.evaluate.assert_called_once()
    actual_settings = cli.evaluate.call_args[0][0]
    assert actual_settings == EvalSettings(
        test_data_path='/tmp/test',
        model_file=DUMMY_MODEL_PATH
    )


def test_handle_rest(mocker):
    m_run = mocker.Mock()
    mocker.patch('qualle.interface.cli.run', m_run)

    cli.handle_rest(
        Namespace(**dict(model=DUMMY_MODEL_PATH, port=[9000], host=['x']))
    )

    m_run.assert_called_once_with(
        RESTSettings(model_file=DUMMY_MODEL_PATH, host='x', port=9000)
    )
