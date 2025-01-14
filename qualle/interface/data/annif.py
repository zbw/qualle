#  Copyright 2021-2025 ZBW  – Leibniz Information Centre for Economics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Any

from qualle.models import TrainData, PredictData

DocumentId = str
Score = float
DocumentQualityEstimation = Tuple[Any, DocumentId]


@dataclass
class AnnifTrainData:
    document_ids: List[DocumentId]
    train_data: TrainData


@dataclass
class AnnifPredictData:
    document_ids: List[DocumentId]
    predict_data: PredictData


_Data = namedtuple(
    "data", ["docs", "predicted_labels", "scores", "true_labels", "doc_ids"]
)


class AnnifHandler:
    def __init__(self, dir: Path):
        self._dir = dir

    def load_train_input(self) -> AnnifTrainData:
        return self._map_to_annif_train_data(self._input_from_annif())

    def load_predict_input(self) -> AnnifPredictData:
        return self._map_to_annif_predict_data(
            self._input_from_annif(include_true_labels=False)
        )

    def store_quality_estimations(
        self, quality_ests: Iterable[DocumentQualityEstimation]
    ):
        for quality_est, doc_id in quality_ests:
            score_fp = self._dir / (doc_id + ".qualle")
            score_fp.write_text(str(quality_est))

    @staticmethod
    def _map_to_annif_train_data(data: _Data) -> AnnifTrainData:
        return AnnifTrainData(
            train_data=TrainData(
                predict_data=AnnifHandler._map_to_predict_data(data),
                true_labels=data.true_labels,
            ),
            document_ids=data.doc_ids,
        )

    @staticmethod
    def _map_to_annif_predict_data(data: _Data) -> AnnifPredictData:
        return AnnifPredictData(
            predict_data=AnnifHandler._map_to_predict_data(data),
            document_ids=data.doc_ids,
        )

    @staticmethod
    def _map_to_predict_data(data: _Data) -> PredictData:
        return PredictData(
            docs=data.docs, predicted_labels=data.predicted_labels, scores=data.scores
        )

    def _input_from_annif(self, include_true_labels=True) -> _Data:
        docs = []
        pred_labels = []
        true_labels = []
        scores = []
        doc_ids = []
        for pred_pth in self._dir.glob("*.annif"):
            doc_id = pred_pth.with_suffix("").name
            doc_ids.append(doc_id)
            with pred_pth.open() as fp:
                scores_for_doc = []
                pred_labels_for_doc = []
                for line in fp.readlines():
                    split = line.rstrip("\n").split("\t")
                    concept_id = self._extract_concept_id_from_annif_label(split[0])
                    pred_labels_for_doc.append(concept_id)
                    score = float(split[2])
                    scores_for_doc.append(score)
                pred_labels.append(pred_labels_for_doc)
                scores.append(scores_for_doc)
            doc_pth = pred_pth.with_suffix(".txt")
            with doc_pth.open() as fp:
                docs.append("".join(list(fp.readlines())))
            label_pth = pred_pth.with_suffix(".tsv")
            if include_true_labels and label_pth.exists():
                with label_pth.open() as fp:
                    true_labels_for_doc = []
                    for line in fp.readlines():
                        split = line.rstrip("\n").split("\t")
                        concept_id = self._extract_concept_id_from_annif_label(split[0])
                        true_labels_for_doc.append(concept_id)
                    true_labels.append(true_labels_for_doc)
        return _Data(
            docs=docs,
            predicted_labels=pred_labels,
            scores=scores,
            true_labels=true_labels,
            doc_ids=doc_ids,
        )

    @staticmethod
    def _extract_concept_id_from_annif_label(uri: str) -> str:
        split = uri.split("/")
        # remove '>' at end of URI
        return split[-1][:-1]
