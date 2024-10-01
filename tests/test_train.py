#  Copyright 2021-2024 ZBW – Leibniz Information Centre for Economics
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

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from qualle.features.confidence import ConfidenceFeatures
from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrator,
    SimpleLabelCalibrationFeatures,
)
from qualle.features.text import TextFeatures
from qualle.models import LabelCalibrationData
from qualle.train import Trainer, FeaturesDataMapper


@pytest.fixture
def l_data(train_data):
    p_data = train_data.predict_data
    return LabelCalibrationData(
        predicted_no_of_labels=np.array([0] * 5),
        predicted_labels=p_data.predicted_labels,
    )


def test_train_trains_qe_pipeline(train_data, mocker):
    t = Trainer(
        train_data=train_data,
        label_calibrator=SimpleLabelCalibrator(ExtraTreesRegressor()),
        quality_regressor=ExtraTreesRegressor(),
        features=[SimpleLabelCalibrationFeatures()],
    )
    spy = mocker.spy(t._qe_p, "train")
    t.train()

    spy.assert_called_once()


def test_features_data_mapper_with_lc(train_data, l_data):
    p_data = train_data.predict_data
    mapper = FeaturesDataMapper(
        set([SimpleLabelCalibrationFeatures, ConfidenceFeatures])
    )
    assert mapper(p_data, l_data) == {
        SimpleLabelCalibrationFeatures: l_data,
        ConfidenceFeatures: p_data.scores,
    }


def test_features_data_mapper_without_lc(train_data, l_data):
    p_data = train_data.predict_data
    mapper = FeaturesDataMapper(set([ConfidenceFeatures]))
    assert mapper(p_data, l_data) == {ConfidenceFeatures: p_data.scores}


def test_features_data_mapper_without_features(train_data, l_data):
    p_data = train_data.predict_data
    mapper = FeaturesDataMapper(set())
    assert mapper(p_data, l_data) == {}


def test_features_data_mapper_for_each_feature(train_data, l_data):
    p_data = train_data.predict_data

    assert FeaturesDataMapper({SimpleLabelCalibrationFeatures})(p_data, l_data) == {
        SimpleLabelCalibrationFeatures: l_data
    }
    assert FeaturesDataMapper({ConfidenceFeatures})(p_data, l_data) == {
        ConfidenceFeatures: p_data.scores
    }
    assert FeaturesDataMapper({TextFeatures})(p_data, l_data) == {
        TextFeatures: p_data.docs
    }
