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
import csv
from pathlib import Path
from typing import List, Callable

from pydantic import BaseModel

from qualle.models import (
    PredictTrainData,
    PredictData,
    LabelCalibrationTrainData,
    Documents,
    Labels,
    Scores,
    Document,
)


class _Data(BaseModel):
    docs: Documents
    predicted_labels: List[Labels]
    scores: List[Scores]
    true_labels: List[Labels]


class _DataRow(BaseModel):
    doc: Document
    predicted_labels: Labels = []
    scores: Scores = []
    true_labels: Labels = []


class RowParseError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def load_predict_train_input(p: Path) -> PredictTrainData:
    return _map_to_predict_train_data(_load_input(p, _parse_predict_train_row))


def load_predict_input(p: Path) -> PredictData:
    return _map_to_predict_data(_load_input(p, _parse_predict_row))


def load_label_calibration_train_input(p: Path) -> LabelCalibrationTrainData:
    return _map_to_label_calibration_train_data(
        _load_input(p, _parse_label_calibration_train_row)
    )


def _map_to_predict_train_data(data: _Data) -> PredictTrainData:
    return PredictTrainData(
        predict_data=_map_to_predict_data(data), true_labels=data.true_labels
    )


def _map_to_predict_data(data: _Data) -> PredictData:
    return PredictData(
        docs=data.docs, predicted_labels=data.predicted_labels, scores=data.scores
    )


def _map_to_label_calibration_train_data(data: _Data) -> LabelCalibrationTrainData:
    return LabelCalibrationTrainData(docs=data.docs, true_labels=data.true_labels)


def _load_input(
    path_to_tsv: Path, row_parser: Callable[[List[str]], _DataRow]
) -> _Data:
    docs = []
    pred_labels = []
    true_labels = []
    scores = []

    with open(path_to_tsv, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            row_data = row_parser(row)
            docs.append(row_data.doc)
            pred_labels.append(row_data.predicted_labels)
            true_labels.append(row_data.true_labels)
            scores.append(row_data.scores)

    return _Data(
        docs=docs, predicted_labels=pred_labels, scores=scores, true_labels=true_labels
    )


def _parse_predict_train_row(row: List[str]) -> _DataRow:
    if len(row) < 3:
        raise RowParseError(
            f"tsv row must have 3 entries, true_labels entry is missing for row {row}"
        )

    predict_row_data = _parse_predict_row(row)

    true_labels_for_row = _parse_true_labels(row[2])
    return _DataRow(
        doc=predict_row_data.doc,
        predicted_labels=predict_row_data.predicted_labels,
        scores=predict_row_data.scores,
        true_labels=true_labels_for_row,
    )


def _parse_predict_row(row: List[str]) -> _DataRow:
    if len(row) < 2:
        raise RowParseError(
            f"tsv row must have 2 entries, "
            f"predicted_labels entry is missing for row {row}"
        )

    doc = row[0]
    pred_labels_for_row = []
    scores_for_row = []
    label_score_pairs = row[1].split(",")
    if len(label_score_pairs) > 0 and label_score_pairs[0]:
        for label_score_pair in label_score_pairs:
            label, score = label_score_pair.split(":")
            pred_labels_for_row.append(label)
            scores_for_row.append(float(score))

    return _DataRow(
        doc=doc, predicted_labels=pred_labels_for_row, scores=scores_for_row
    )


def _parse_label_calibration_train_row(row: List[str]) -> _DataRow:
    if len(row) < 2:
        raise RowParseError(
            f"tsv row must have 2 entries, true_labels entry is missing for row {row}"
        )

    doc = row[0]
    true_labels = _parse_true_labels(row[1])

    return _DataRow(doc=doc, true_labels=true_labels)


def _parse_true_labels(row_entry: str) -> Labels:
    return list(filter(bool, row_entry.split(",")))
