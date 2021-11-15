#  Copyright 2021 ZBW – Leibniz Information Centre for Economics
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
import json
import logging

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from qualle.interface import internal
from qualle.interface.config import TrainSettings, FeaturesEnum, \
    RegressorSettings, EvalSettings, RESTSettings
from qualle.interface.rest import create_app, PREDICT_ENDPOINT, Documents, \
    Document, QualityEstimation, QualityScores, Metric


@pytest.fixture
def train_data_file(tmp_path):
    return tmp_path / 'test_data.tsv'


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / 'output.model'


def test_train_stores_model(train_data_file, model_path):
    train(train_data_file, model_path)
    assert model_path.is_file()


def test_eval_prints_scores(train_data_file, model_path, caplog):
    caplog.set_level(logging.INFO)

    train(train_data_file, model_path)

    settings = EvalSettings(
        test_data_path=train_data_file,
        model_file=model_path
    )
    internal.evaluate(settings)

    assert 'Scores:' in caplog.text

    # We can make following expectations because
    # test_data_file == train_data_file
    assert 'mean_squared_error: 0' in caplog.text
    assert 'explained_variance_score: 1' in caplog.text
    assert 'correlation_coefficient: nan' in caplog.text


def test_rest(train_data_file, model_path):
    train(train_data_file, model_path)
    settings = RESTSettings(
        model_file=model_path
    )
    app = create_app(settings)
    client = TestClient(app)
    res = client.post(PREDICT_ENDPOINT, json=Documents(documents=[
        Document(
            content='doc', predicted_labels=['concept0', 'concept1'],
            scores=[0.5, 1])]
    ).dict())

    assert res.status_code == status.HTTP_200_OK
    # We can make following assumption due to the construction of train data
    assert res.json() == json.loads(QualityEstimation(scores=[
        QualityScores(name=Metric.RECALL, scores=[1])]).json())


def train(train_data_file, output_path):
    with train_data_file.open('w') as f:
        f.write(
            'doc\tconcept0:0.5,concept1:1\tconcept0\n' * 100
        )
    settings = TrainSettings(
        label_calibrator_regressor=RegressorSettings(
            regressor_class='sklearn.ensemble.GradientBoostingRegressor',
            params={}
        ),
        quality_estimator_regressor=RegressorSettings(
            regressor_class='sklearn.ensemble.GradientBoostingRegressor',
            params={}
        ),
        train_data_path=train_data_file,
        output_path=output_path,
        features=[FeaturesEnum.TEXT],
    )

    internal.train(settings)
