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
import csv
from collections import namedtuple
from pathlib import Path

from qualle.models import TrainData, PredictData

_Data = namedtuple("data", ["docs", "predicted_labels", "scores", "true_labels"])


def load_train_input(p: Path) -> TrainData:
    return _map_to_train_data(_load_input(p))


def load_predict_input(p: Path):
    return _map_to_predict_data(_load_input(p, include_true_labels=False))


def _map_to_train_data(data: _Data) -> TrainData:
    return TrainData(
        predict_data=_map_to_predict_data(data), true_labels=data.true_labels
    )


def _map_to_predict_data(data: _Data) -> PredictData:
    return PredictData(
        docs=data.docs, predicted_labels=data.predicted_labels, scores=data.scores
    )


def _load_input(path_to_tsv: Path, include_true_labels=True) -> _Data:
    docs = []
    pred_labels = []
    true_labels = []
    scores = []

    with open(path_to_tsv, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            docs.append(row[0])
            pred_labels_for_row = []
            scores_for_row = []
            label_score_pairs = row[1].split(",")
            if len(label_score_pairs) > 0 and label_score_pairs[0]:
                for label_score_pair in label_score_pairs:
                    label, score = label_score_pair.split(":")
                    pred_labels_for_row.append(label)
                    scores_for_row.append(float(score))
            pred_labels.append(pred_labels_for_row)
            scores.append(scores_for_row)

            if include_true_labels and len(row) > 2:
                true_labels.append(list(filter(bool, row[2].split(","))))

    return _Data(
        docs=docs, predicted_labels=pred_labels, scores=scores, true_labels=true_labels
    )
