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

from enum import Enum
from functools import lru_cache
from typing import List, Optional

import uvicorn
from fastapi import status, FastAPI, Depends, APIRouter
from pydantic.main import BaseModel

from qualle.interface.config import RESTSettings
from qualle.interface.internal import load_model as internal_load_model
from qualle.models import PredictData
from qualle.pipeline import QualityEstimationPipeline

PREDICT_ENDPOINT = '/predict'


class Document(BaseModel):
    content: str
    predicted_labels: List[str]
    scores: List[float]


class Documents(BaseModel):
    documents: List[Document]


class Metric(Enum):
    RECALL = 'recall'


class QualityScores(BaseModel):
    name: Metric
    scores: List[float]


class QualityEstimation(BaseModel):

    scores: List[QualityScores]


router = APIRouter()


@lru_cache
def load_model() -> QualityEstimationPipeline:
    settings = RESTSettings()
    return internal_load_model(str(settings.model_file))


@router.post(
    PREDICT_ENDPOINT, status_code=status.HTTP_200_OK,
    response_model=QualityEstimation
)
def predict(
        documents: Documents,
        qe_pipeline: QualityEstimationPipeline = Depends(load_model)
) -> QualityEstimation:
    predict_data = _map_documents_to_predict_data(documents)
    scores = qe_pipeline.predict(predict_data)
    return QualityEstimation(scores=[QualityScores(
        name=Metric.RECALL, scores=list(scores))
    ])


@router.get('/_up')
def up():
    return True


def _map_documents_to_predict_data(documents: Documents) -> PredictData:
    docs = []
    p_labels = []
    scores = []

    for idx, doc in enumerate(documents.documents):
        docs.append(doc.content)
        p_labels.append(doc.predicted_labels)
        scores.append(doc.scores)
    return PredictData(
        docs=docs, predicted_labels=p_labels, scores=scores
    )


def create_app(settings: Optional[RESTSettings] = None):
    settings = settings or RESTSettings()
    app = FastAPI()
    app.include_router(router)
    m = internal_load_model(str(settings.model_file))
    app.dependency_overrides[load_model] = lambda: m

    return app


def run(settings: Optional[RESTSettings] = None):
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
