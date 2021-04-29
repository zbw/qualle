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
        test_data_file=train_data_file,
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
        train_data_file=train_data_file,
        output_path=output_path,
        features=[FeaturesEnum.TEXT],
    )

    internal.train(settings)
