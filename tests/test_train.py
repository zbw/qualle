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

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from qualle.features.confidence import ConfidenceFeatures
from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrator, SimpleLabelCalibrationFeatures
from qualle.models import LabelCalibrationData
from qualle.train import Trainer, FeaturesDataMapper


@pytest.fixture
def l_data(train_data):
    p_data = train_data.predict_data
    return LabelCalibrationData(
        predicted_no_of_labels=np.array([0] * 5),
        predicted_labels=p_data.predicted_labels
    )


def test_train_trains_qe_pipeline(train_data, mocker):
    t = Trainer(
        train_data=train_data,
        label_calibrator=SimpleLabelCalibrator(ExtraTreesRegressor()),
        recall_predictor_regressor=ExtraTreesRegressor(),
        features=[SimpleLabelCalibrationFeatures()]
    )
    spy = mocker.spy(t._qe_p, 'train')
    t.train()

    spy.assert_called_once()


def test_features_data_mapper_with_lc(train_data, l_data):
    p_data = train_data.predict_data
    mapper = FeaturesDataMapper(set([
        SimpleLabelCalibrationFeatures, ConfidenceFeatures])
    )
    assert mapper(p_data, l_data) == {
        SimpleLabelCalibrationFeatures: l_data,
        ConfidenceFeatures: p_data.scores
    }


def test_features_data_mapper_without_lc(train_data, l_data):
    p_data = train_data.predict_data
    mapper = FeaturesDataMapper(set([ConfidenceFeatures]))
    assert mapper(p_data, l_data) == {
        ConfidenceFeatures: p_data.scores
    }


def test_features_data_mapper_without_features(train_data, l_data):
    p_data = train_data.predict_data
    mapper = FeaturesDataMapper(set())
    assert mapper(p_data, l_data) == {}
