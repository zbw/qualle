#  Copyright 2021-2023 ZBW  – Leibniz Information Centre for Economics
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
import os
from glob import glob
from pathlib import Path

from qualle.interface.io.common import Data, \
    map_to_train_data, map_to_predict_data
from qualle.models import TrainData, PredictData


class AnnifHandler:

    def __init__(self, dir: Path):
        self._dir = dir

    def load_train_input(self) -> TrainData:
        return map_to_train_data(self._input_from_annif())

    def load_predict_input(self) -> PredictData:
        return map_to_predict_data(self._input_from_annif(
            include_true_labels=False)
        )

    def _input_from_annif(self, include_true_labels=True) -> Data:
        docs = []
        pred_labels = []
        true_labels = []
        scores = []
        for pred_pth in glob((os.path.join(self._dir, '*.annif'))):
            with open(pred_pth) as fp:
                scores_for_doc = []
                pred_labels_for_doc = []
                for line in fp.readlines():
                    split = line.rstrip('\n').split('\t')
                    concept_id =\
                        self._extract_concept_id_from_annif_label(split[0])
                    pred_labels_for_doc.append(concept_id)
                    score = float(split[2])
                    scores_for_doc.append(score)
                pred_labels.append(pred_labels_for_doc)
                scores.append(scores_for_doc)
            doc_pth = pred_pth.replace('.annif', '.txt')
            with open(doc_pth) as fp:
                docs.append(''.join(list(fp.readlines())))
            label_pth = pred_pth.replace('.annif', '.tsv')
            if include_true_labels and os.path.exists(label_pth):
                with open(label_pth) as fp:
                    true_labels_for_doc = []
                    for line in fp.readlines():
                        split = line.rstrip('\n').split('\t')
                        concept_id =\
                            self._extract_concept_id_from_annif_label(split[0])
                        true_labels_for_doc.append(concept_id)
                    true_labels.append(true_labels_for_doc)
        return Data(
            docs=docs, predicted_labels=pred_labels,
            scores=scores, true_labels=true_labels
        )

    @staticmethod
    def _extract_concept_id_from_annif_label(uri: str) -> str:
        split = uri.split('/')
        # remove '>' at end of URI
        return split[-1][:-1]
