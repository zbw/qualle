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
import logging

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from qualle.features.combined import CombinedFeatures
from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrationFeatures,
    SimpleLabelCalibrator,
)
from qualle.models import PredictData, TrainData
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import QualityEstimator


@pytest.fixture
def qp(mocker):
    label_calibrator = SimpleLabelCalibrator(ExtraTreesRegressor())
    recall_predictor = QualityEstimator(
        regressor=ExtraTreesRegressor(),
        features=CombinedFeatures([SimpleLabelCalibrationFeatures()]),
    )
    mocker.patch("qualle.pipeline.cross_val_predict", mocker.Mock(return_value=[1] * 5))
    return QualityEstimationPipeline(
        label_calibrator=label_calibrator,
        recall_predictor=recall_predictor,
        features_data_mapper=lambda _, l_data: {SimpleLabelCalibrationFeatures: l_data},
    )


@pytest.fixture
def train_data():
    labels = [["c"] for _ in range(5)]
    return TrainData(
        predict_data=PredictData(
            docs=[f"d{i}" for i in range(5)], predicted_labels=labels, scores=[[0]] * 5
        ),
        true_labels=labels,
    )


@pytest.fixture
def train_data_with_some_empty_labels(train_data):
    train_data.predict_data.predicted_labels = [["c"], [], ["c"], [], ["c"]]

    return train_data


@pytest.fixture
def train_data_with_all_empty_labels(train_data):
    train_data.predict_data.predicted_labels = [[]] * 5

    return train_data


def test_train(qp, train_data, mocker):
    calibrator = qp._label_calibrator
    mocker.spy(calibrator, "fit")
    mocker.spy(qp._recall_predictor, "fit")

    qp.train(train_data)

    calibrator.fit.assert_called()
    qp._recall_predictor.fit.assert_called_once()

    actual_lc_docs = calibrator.fit.call_args[0][0]
    actual_lc_true_labels = calibrator.fit.call_args[0][1]

    assert actual_lc_docs == train_data.predict_data.docs
    assert actual_lc_true_labels == train_data.true_labels

    features_data = qp._recall_predictor.fit.call_args[0][0]
    actual_true_recall = qp._recall_predictor.fit.call_args[0][1]

    assert SimpleLabelCalibrationFeatures in features_data
    actual_label_calibration_data = features_data[SimpleLabelCalibrationFeatures]
    # Because of how our input data is designed,
    # we can make following assertions
    only_ones = [1] * 5
    assert actual_label_calibration_data.predicted_no_of_labels == only_ones
    assert actual_label_calibration_data.predicted_labels == [["c"]] * 5
    assert actual_true_recall == only_ones


def test_predict(qp, train_data):
    p_data = train_data.predict_data

    qp.train(train_data)

    # Because of how our input data is designed,
    # we can make following assertion
    assert np.array_equal(qp.predict(p_data), [1] * 5)


def test_predict_with_some_empty_labels_returns_zero_recall(
    qp, train_data_with_some_empty_labels
):
    p_data = train_data_with_some_empty_labels.predict_data

    qp.train(train_data_with_some_empty_labels)

    assert np.array_equal(qp.predict(p_data), [1, 0, 1, 0, 1])


def test_predict_with_all_empty_labels_returns_only_zero_recall(
    qp, train_data_with_all_empty_labels
):
    p_data = train_data_with_all_empty_labels.predict_data

    qp.train(train_data_with_all_empty_labels)

    assert np.array_equal(qp.predict(p_data), [0] * 5)


def test_debug_prints_time_if_activated(qp, caplog):
    qp._should_debug = True
    caplog.set_level(logging.DEBUG)
    with qp._debug("msg"):
        _ = 1 + 1
    assert "Ran msg in " in caplog.text
    assert "seconds" in caplog.text


def test_debug_prints_nothing_if_not_activated(qp, caplog):
    caplog.set_level(logging.DEBUG)
    with qp._debug("msg"):
        _ = 1 + 1
    assert "" == caplog.text
