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
from pathlib import Path

import pytest
from rdflib import URIRef
from sklearn.ensemble import GradientBoostingRegressor

from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrator, SimpleLabelCalibrationFeatures
from qualle.features.label_calibration.thesauri_label_calibration import \
    ThesauriLabelCalibrator, ThesauriLabelCalibrationFeatures
from qualle.features.text import TextFeatures
from qualle.interface.config import TrainSettings, FeaturesEnum, \
    RegressorSettings, SubthesauriLabelCalibrationSettings, EvalSettings
import qualle.interface.internal as internal
from qualle.interface.internal import train

import tests.interface.common as c


@pytest.fixture(autouse=True)
def mock_io(mocker, train_data):
    mocker.patch('qualle.interface.internal.dump')
    mocker.patch('qualle.interface.internal.load')
    mocker.patch(
        'qualle.interface.internal.train_input_from_tsv',
        mocker.Mock(return_value=train_data)
    )


@pytest.fixture
def train_settings():
    return TrainSettings(
        train_data_file='/tmp/train',
        output_path='/tmp/output',
        should_debug=False,
        features=[FeaturesEnum.TEXT],
        label_calibrator_regressor=RegressorSettings(
            regressor_class='sklearn.ensemble.GradientBoostingRegressor',
            params=dict(n_estimators=10, max_depth=8)
        ),
        quality_estimator_regressor=RegressorSettings(
            regressor_class='sklearn.ensemble.GradientBoostingRegressor',
            params=dict(n_estimators=10, max_depth=8)
        ),
    )


def test_train_trains_trainer(train_settings, mocker):
    m_trainer = mocker.Mock()
    m_trainer_cls = mocker.Mock(return_value=m_trainer)
    m_trainer.train = mocker.Mock(return_value='testmodel')
    mocker.patch('qualle.interface.internal.Trainer', m_trainer_cls)
    train(train_settings)

    m_trainer.train.assert_called_once()
    internal.dump.assert_called_once_with('testmodel', Path('/tmp/output'))


def test_train_without_slc_creates_respective_trainer(train_settings, mocker,
                                                      train_data):
    mocker.patch('qualle.interface.internal.Trainer')

    train(train_settings)

    internal.Trainer.assert_called_once()
    call_args = internal.Trainer.call_args[1]
    assert call_args.get('train_data') == train_data
    lc = call_args.get('label_calibrator')
    assert isinstance(lc, SimpleLabelCalibrator)
    assert lc.regressor.__dict__ == GradientBoostingRegressor(
        **dict(n_estimators=10, max_depth=8)).__dict__
    qr = call_args.get('quality_regressor')
    assert isinstance(qr, GradientBoostingRegressor)
    assert qr.__dict__ == GradientBoostingRegressor(
        **dict(n_estimators=10, max_depth=8)).__dict__
    assert list(map(lambda f: f.__class__, call_args.get('features'))) == [
        TextFeatures, SimpleLabelCalibrationFeatures]
    assert call_args.get('should_debug') is False


def test_train_with_slc_creates_respective_trainer(
        train_settings, mocker, train_data
):
    m_graph = mocker.Mock()
    m_graph_cls = mocker.Mock(return_value=m_graph)
    mocker.patch('qualle.interface.internal.Graph', m_graph_cls)
    m_lcfst = mocker.Mock()
    m_lcfst_cls = mocker.Mock(return_value=m_lcfst)
    mocker.patch(
        'qualle.interface.internal.LabelCountForSubthesauriTransformer',
        m_lcfst_cls
    )
    mocker.patch('qualle.interface.internal.Trainer')

    train_settings.subthesauri_label_calibration = \
        SubthesauriLabelCalibrationSettings(
            thesaurus_file=c.DUMMY_THESAURUS_FILE,
            subthesaurus_type=c.DUMMY_SUBTHESAURUS_TYPE,
            concept_type=c.DUMMY_CONCEPT_TYPE,
            concept_type_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX,
            subthesauri=[c.DUMMY_SUBTHESAURUS_A, c.DUMMY_SUBTHESAURUS_B]
        )

    train(train_settings)

    m_graph_cls.assert_called_once()
    m_graph.parse.assert_called_once_with(c.DUMMY_THESAURUS_FILE)

    m_lcfst_cls.assert_called_once_with(
        graph=m_graph,
        subthesaurus_type_uri=URIRef(
            c.DUMMY_SUBTHESAURUS_TYPE),
        concept_type_uri=URIRef(
            c.DUMMY_CONCEPT_TYPE),
        subthesauri=[
            URIRef(c.DUMMY_SUBTHESAURUS_A),
            URIRef(c.DUMMY_SUBTHESAURUS_B)],
        concept_uri_prefix=c.DUMMY_CONCEPT_TYPE_PREFIX
    )
    m_lcfst.fit.assert_called_once()

    internal.Trainer.assert_called_once()

    call_args = internal.Trainer.call_args[1]
    assert call_args.get('train_data') == train_data
    lc = call_args.get('label_calibrator')
    assert isinstance(lc, ThesauriLabelCalibrator)
    assert lc.regressor_class == GradientBoostingRegressor
    assert lc.regressor_params == dict(n_estimators=10, max_depth=8)
    qr = call_args.get('quality_regressor')
    assert isinstance(qr, GradientBoostingRegressor)
    assert qr.__dict__ == GradientBoostingRegressor(
        **dict(n_estimators=10, max_depth=8)).__dict__
    assert list(map(
        lambda f: f.__class__, call_args.get('features'))) == [
               TextFeatures, ThesauriLabelCalibrationFeatures]
    assert call_args.get('should_debug') is False


def test_evaluate(mocker, train_data):
    m_eval = mocker.Mock()
    m_eval.evaluate.return_value = {}
    m_eval_cls = mocker.Mock(return_value=m_eval)
    mocker.patch('qualle.interface.internal.Evaluator', m_eval_cls)
    internal.load.return_value = 'testmodel'

    settings = EvalSettings(
        test_data_file='/tmp/test',
        model_file='/tmp/model'
    )
    internal.evaluate(settings)

    internal.train_input_from_tsv.assert_called_once_with('/tmp/test')

    m_eval_cls.assert_called_once_with(
        train_data,
        'testmodel'
    )
    m_eval.evaluate.assert_called_once()
