#  Copyright 2021-2023 ZBW – Leibniz Information Centre for Economics
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
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from qualle.evaluate import Evaluator, scores
from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrationFeatures,
    SimpleLabelCalibrator,
)
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import QualityEstimator


@pytest.fixture
def evaluator(train_data):
    qe_p = QualityEstimationPipeline(
        recall_predictor=QualityEstimator(
            regressor=ExtraTreesRegressor(), features=SimpleLabelCalibrationFeatures()
        ),
        label_calibrator=SimpleLabelCalibrator(ExtraTreesRegressor()),
        features_data_mapper=lambda _, l_data: l_data,
    )
    qe_p.train(train_data)
    return Evaluator(train_data.predict_split, qe_p)


def test_evaluate_returns_scores(evaluator):
    scores = evaluator.evaluate()

    assert {
        "explained_variance_score",
        "mean_squared_error",
        "correlation_coefficient",
    } == set(scores.keys())


def test_scores():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    s = scores(y_true, y_pred)

    assert s["explained_variance_score"] == 0.9571734475374732
    assert s["mean_squared_error"] == 0.375
    assert s["correlation_coefficient"] == 0.9848696184482703
