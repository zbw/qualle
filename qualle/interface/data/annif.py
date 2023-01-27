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
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Any

from qualle.models import PredictData, PredictTrainData, LabelCalibrationTrainData

DocumentId = str
Score = float
DocumentQualityEstimation = Tuple[Any, DocumentId]


@dataclass
class AnnifPredictTrainData:
    document_ids: List[DocumentId]
    predict_train_data: PredictTrainData


@dataclass
class AnnifPredictData:
    document_ids: List[DocumentId]
    predict_data: PredictData


@dataclass
class AnnifLabelCalibrationTrainData:
    document_ids: List[DocumentId]
    label_calibration_train_data: LabelCalibrationTrainData


class AnnifLoadError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


_Data = namedtuple(
    "data", ["docs", "predicted_labels", "scores", "true_labels", "doc_ids"]
)


def load_predict_train_input(dir: Path) -> AnnifPredictTrainData:
    parser_config = _ParserConfig(
        use_doc_parser=True, use_true_label_parser=True, use_prediction_parser=True
    )
    return _map_to_annif_predict_train_data(
        _load_input(dir=dir, parser_config=parser_config)
    )


def load_predict_input(dir: Path) -> AnnifPredictData:
    parser_config = _ParserConfig(
        use_doc_parser=True, use_true_label_parser=False, use_prediction_parser=True
    )
    return _map_to_annif_predict_data(_load_input(dir=dir, parser_config=parser_config))


def load_label_calibration_train_input(dir: Path) -> AnnifLabelCalibrationTrainData:
    parser_config = _ParserConfig(
        use_doc_parser=True, use_true_label_parser=True, use_prediction_parser=False
    )
    return _map_to_annif_label_calibration_train_data(
        _load_input(dir=dir, parser_config=parser_config)
    )


def store_quality_estimations(
    dir: Path, quality_ests: Iterable[DocumentQualityEstimation]
):
    for quality_est, doc_id in quality_ests:
        score_fp = dir / (doc_id + ".qualle")
        score_fp.write_text(str(quality_est))


@dataclass
class _ParserConfig:
    use_doc_parser: bool
    use_prediction_parser: bool
    use_true_label_parser: bool

    def get_parsers(self):
        parsers = []
        if self.use_doc_parser:
            parsers.append(_DocParser())
        if self.use_prediction_parser:
            parsers.append(_PredictionParser())
        if self.use_true_label_parser:
            parsers.append(_TrueLabelsParser())
        return parsers


class _DocParser:
    FILE_EXT = "txt"

    @staticmethod
    def parse(p: Path) -> str:
        with p.open() as fp:
            return "".join(list(fp.readlines()))


class _PredictionParser:
    FILE_EXT = "annif"

    @staticmethod
    def parse(p: Path):
        with p.open() as fp:
            scores_for_doc = []
            pred_labels_for_doc = []
            for line in fp.readlines():
                split = line.rstrip("\n").split("\t")
                concept_id = _extract_concept_id_from_annif_label(split[0])
                pred_labels_for_doc.append(concept_id)
                score = float(split[2])
                scores_for_doc.append(score)
            return pred_labels_for_doc, scores_for_doc


class _TrueLabelsParser:

    FILE_EXT = "tsv"

    @staticmethod
    def parse(p: Path):
        with p.open() as fp:
            true_labels_for_doc = []
            for line in fp.readlines():
                split = line.rstrip("\n").split("\t")
                concept_id = _extract_concept_id_from_annif_label(split[0])
                true_labels_for_doc.append(concept_id)
            return true_labels_for_doc


def _map_to_annif_predict_train_data(data: _Data) -> AnnifPredictTrainData:
    return AnnifPredictTrainData(
        predict_train_data=PredictTrainData(
            predict_data=_map_to_predict_data(data),
            true_labels=data.true_labels,
        ),
        document_ids=data.doc_ids,
    )


def _map_to_annif_predict_data(data: _Data) -> AnnifPredictData:
    return AnnifPredictData(
        predict_data=_map_to_predict_data(data),
        document_ids=data.doc_ids,
    )


def _map_to_annif_label_calibration_train_data(
    data: _Data,
) -> AnnifLabelCalibrationTrainData:
    return AnnifLabelCalibrationTrainData(
        label_calibration_train_data=LabelCalibrationTrainData(
            docs=data.docs,
            true_labels=data.true_labels,
        ),
        document_ids=data.doc_ids,
    )


def _map_to_predict_data(data: _Data) -> PredictData:
    return PredictData(
        docs=data.docs, predicted_labels=data.predicted_labels, scores=data.scores
    )


def _load_input(dir: Path, parser_config: _ParserConfig) -> _Data:
    docs = []
    pred_labels = []
    true_labels = []
    scores = []
    doc_ids = []

    parsers = parser_config.get_parsers()

    for p in dir.glob(f"*.{parsers[0].FILE_EXT}"):
        doc_id = p.with_suffix("").name
        doc_ids.append(doc_id)

        for parser in parsers:
            file_path = p.with_suffix(f".{parser.FILE_EXT}")
            if not file_path.exists():
                raise AnnifLoadError(f"file {file_path} is missing")
            parse_result = parser.parse(file_path)
            if isinstance(parser, _DocParser):
                docs.append(parse_result)
            if isinstance(parser, _PredictionParser):
                pred_labels_for_doc, scores_for_doc = parse_result
                pred_labels.append(pred_labels_for_doc)
                scores.append(scores_for_doc)
            if isinstance(parser, _TrueLabelsParser):
                true_labels.append(parse_result)
    return _Data(
        docs=docs,
        predicted_labels=pred_labels,
        scores=scores,
        true_labels=true_labels,
        doc_ids=doc_ids,
    )


def _extract_concept_id_from_annif_label(uri: str) -> str:
    split = uri.split("/")
    # remove '>' at end of URI
    return split[-1][:-1]
