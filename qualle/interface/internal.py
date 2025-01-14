#  Copyright 2021-2025 ZBW – Leibniz Information Centre for Economics
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

"""Internal interface to access Qualle functionality"""

from importlib import import_module
from pathlib import Path
from typing import Type, Optional, List, Any, Union

from joblib import dump, load
from rdflib import Graph, URIRef

from qualle.evaluate import Evaluator
from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrator,
    SimpleLabelCalibrationFeatures,
)
from qualle.features.label_calibration.thesauri_label_calibration import (
    ThesauriLabelCalibrator,
    ThesauriLabelCalibrationFeatures,
    LabelCountForSubthesauriTransformer,
    Thesaurus,
)
from qualle.interface.config import TrainSettings, EvalSettings, PredictSettings
from qualle.models import TrainData, PredictData
from qualle.train import Trainer
from qualle.utils import get_logger, timeit
from qualle.interface.data.annif import AnnifHandler
from qualle.interface.data.tsv import (
    load_train_input as load_tsv_train_input,
    load_predict_input,
)


def train(settings: TrainSettings):
    logger = get_logger()
    path_to_train_data = settings.train_data_path
    path_to_model_output_file = settings.output_path
    slc_settings = settings.subthesauri_label_calibration
    features = list(map(lambda f: f.value(), settings.features))

    lc_regressor_cls = _get_class_from_str(
        settings.label_calibrator_regressor.regressor_class
    )
    lc_regressor_params = settings.label_calibrator_regressor.params
    logger.debug(
        "Use (%s %s) as Regressor for Label Calibration",
        settings.label_calibrator_regressor.regressor_class,
        settings.label_calibrator_regressor.params,
    )

    quality_estimator_cls = _get_class_from_str(
        settings.quality_estimator_regressor.regressor_class
    )
    quality_estimator = quality_estimator_cls(
        **settings.quality_estimator_regressor.params
    )
    logger.debug(
        "Use (%s %s) as Regressor for Quality Estimation",
        settings.quality_estimator_regressor.regressor_class,
        settings.quality_estimator_regressor.params,
    )

    with timeit() as t:
        train_data = _load_train_input(path_to_train_data)

    logger.debug("Loaded train data in %.4f seconds", t())

    if slc_settings:
        logger.info("Run training with Subthesauri Label Calibration")
        path_to_graph = slc_settings.thesaurus_file
        g = Graph()
        with timeit() as t:
            g.parse(path_to_graph)
        logger.debug("Parsed RDF Graph in %.4f seconds", t())

        thesaurus = Thesaurus(
            graph=g,
            subthesaurus_type_uri=URIRef(slc_settings.subthesaurus_type),
            concept_type_uri=URIRef(slc_settings.concept_type),
            concept_uri_prefix=slc_settings.concept_type_prefix,
        )
        subthesauri = list(map(lambda s: URIRef(s), slc_settings.subthesauri))
        if not subthesauri:
            logger.info("No subthesauri passed, will extract subthesauri by type")
            subthesauri = thesaurus.get_all_subthesauri()
            logger.debug("Extracted %d subthesauri", len(subthesauri))
        if slc_settings.use_sparse_count_matrix:
            logger.info(
                "Will use Sparse Count Matrix for " "Subthesauri Label Calibration"
            )
        transformer = LabelCountForSubthesauriTransformer(
            use_sparse_count_matrix=slc_settings.use_sparse_count_matrix
        )
        transformer.init(thesaurus=thesaurus, subthesauri=subthesauri)

        label_calibration_features = ThesauriLabelCalibrationFeatures(
            transformer=transformer
        )
        features.append(label_calibration_features)
        t = Trainer(
            train_data=train_data,
            label_calibrator=ThesauriLabelCalibrator(
                regressor_class=lc_regressor_cls,
                regressor_params=lc_regressor_params,
                transformer=transformer,
            ),
            quality_regressor=quality_estimator,
            features=features,
            should_debug=settings.should_debug,
        )
    else:
        logger.info("Run training with Simple Label Calibration")
        label_calibration_features = SimpleLabelCalibrationFeatures()
        features.append(label_calibration_features)
        t = Trainer(
            train_data=train_data,
            label_calibrator=SimpleLabelCalibrator(
                lc_regressor_cls(**lc_regressor_params)
            ),
            quality_regressor=quality_estimator,
            features=features,
            should_debug=settings.should_debug,
        )

    model = t.train()
    logger.info("Store trained model in %s", path_to_model_output_file)
    dump(model, path_to_model_output_file)


def _get_class_from_str(fully_qualified_path: str) -> Type:
    split = fully_qualified_path.split(".")
    module = ".".join(split[:-1])
    cls_name = split[-1]
    return getattr(import_module(module), cls_name)


def evaluate(settings: EvalSettings):
    logger = get_logger()
    path_to_test_data = settings.test_data_path
    path_to_model_file = settings.mdl_file
    model = load_model(str(path_to_model_file))
    logger.info("Run evaluation with model:\n%s", model)
    test_input = _load_train_input(path_to_test_data)
    ev = Evaluator(test_input, model)
    eval_data = ev.evaluate()
    logger.info("\nScores:")
    for metric, score in eval_data.items():
        logger.info(f"{metric}: {score}")


def predict(settings: PredictSettings):
    logger = get_logger()
    path_to_predict_data = settings.predict_data_path
    path_to_model_file = settings.mdl_file
    output_path = settings.output_path
    model = load_model(str(path_to_model_file))
    io_handler = _get_predict_io_handler(
        predict_data_path=path_to_predict_data, output_path=output_path
    )
    predict_data = io_handler.load_predict_data()
    logger.info("Run predict with model:\n%s", model)

    scores = model.predict(predict_data)
    io_handler.store(scores)


def load_model(path_to_model_file: str):
    return load(path_to_model_file)


def _load_train_input(p: Path) -> TrainData:
    if p.is_dir():
        train_input = AnnifHandler(dir=p).load_train_input().train_data
    else:
        train_input = load_tsv_train_input(p)
    return train_input


class PredictTSVIOHandler:
    def __init__(self, predict_data_path: Path, output_path: Optional[Path] = None):
        self._predict_data_path = predict_data_path
        self._output_path = output_path

    def load_predict_data(self):
        return load_predict_input(self._predict_data_path)

    def store(self, data: List[Any]):
        with self._output_path.open("w") as f:
            for d in data:
                f.write(str(d) + "\n")


class PredictAnnifIOHandler:
    def __init__(self, predict_data_path: Path):
        self._annif_handler = AnnifHandler(dir=predict_data_path)
        self._doc_ids = []

    def load_predict_data(self) -> PredictData:
        data = self._annif_handler.load_predict_input()
        self._doc_ids = data.document_ids
        return data.predict_data

    def store(self, data: List[Any]):
        quality_ests = zip(data, self._doc_ids)
        self._annif_handler.store_quality_estimations(quality_ests)


def _get_predict_io_handler(
    predict_data_path: Path, output_path: Optional[Path]
) -> Union[PredictAnnifIOHandler, PredictTSVIOHandler]:
    if predict_data_path.is_dir():
        return PredictAnnifIOHandler(predict_data_path)
    return PredictTSVIOHandler(
        predict_data_path=predict_data_path, output_path=output_path
    )
