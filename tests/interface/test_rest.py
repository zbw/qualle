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

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from qualle.interface.config import RESTSettings
from qualle.interface.rest import Document, QualityScores, \
    QualityEstimation, Metric, \
    _map_documents_to_predict_data, Documents, create_app, PREDICT_ENDPOINT, \
    run


@pytest.fixture
def mocked_pipeline(mocker):
    m_pipe = mocker.Mock()
    m_pipe.predict.side_effect = lambda p_data: list(range(len(p_data.scores)))

    m_load_model = mocker.Mock(return_value=m_pipe)
    mocker.patch('qualle.interface.rest.internal_load_model', m_load_model)

    return m_pipe


@pytest.fixture
def client(mocked_pipeline):
    app = create_app(RESTSettings(model_file='/tmp/dummy'))
    client = TestClient(app)
    return client


@pytest.fixture
def documents(train_data):
    p_data = train_data.predict_data
    return Documents(documents=[
        Document(
            content=p_data.docs[idx],
            predicted_labels=p_data.predicted_labels[idx],
            scores=p_data.scores[idx]
        ) for idx in range(len(p_data.docs))
    ])


def test_return_http_200_for_predict(client, documents):
    resp = client.post(PREDICT_ENDPOINT, json=documents.dict())
    assert resp.status_code == status.HTTP_200_OK


def test_return_scores_for_predict(client, documents):
    resp = client.post(PREDICT_ENDPOINT, json=documents.dict())

    expected_scores = QualityEstimation(
        scores=[QualityScores(
            name=Metric.RECALL,
            scores=list(range(len(documents.documents)))
        )]
    )
    assert resp.json() == json.loads(expected_scores.json())


def test_return_http_200_for_up(client):
    resp = client.get('/_up')
    assert resp.status_code == status.HTTP_200_OK


def test_run(mocker):
    m_app = mocker.Mock()
    m_create_app = mocker.Mock(return_value=m_app)
    mocker.patch('qualle.interface.rest.create_app', m_create_app)
    m_uvicorn_run = mocker.Mock()
    mocker.patch('qualle.interface.rest.uvicorn.run', m_uvicorn_run)

    settings = RESTSettings(model_file='/tmp/model')

    run(settings)

    m_create_app.assert_called_once_with(settings)
    m_uvicorn_run.assert_called_once_with(
        m_app, host=settings.host, port=settings.port
    )


def test_map_documents_to_predict_data(documents, train_data):
    assert _map_documents_to_predict_data(documents) == train_data.predict_data
