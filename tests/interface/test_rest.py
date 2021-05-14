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
