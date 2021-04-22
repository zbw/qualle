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

from importlib import import_module
from typing import Type

from joblib import dump, load
from rdflib import Graph, URIRef

from qualle.evaluate import Evaluator
from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrator, SimpleLabelCalibrationFeatures
from qualle.features.label_calibration.thesauri_label_calibration import \
    ThesauriLabelCalibrator, ThesauriLabelCalibrationFeatures, \
    LabelCountForSubthesauriTransformer
from qualle.interface.config import TrainSettings, EvalSettings
from qualle.train import Trainer
from qualle.utils import get_logger, train_input_from_tsv, timeit


def train(settings: TrainSettings):
    logger = get_logger()
    path_to_train_tsv = settings.train_data_file
    path_to_model_output_file = settings.output_path
    train_data = train_input_from_tsv(str(path_to_train_tsv))
    slc_settings = settings.subthesauri_label_calibration
    features = list(map(lambda f: f.value(), settings.features))

    quality_estimator_cls = _get_class_from_str(
        settings.quality_estimator_regressor.regressor_class
    )
    quality_estimator = quality_estimator_cls(
        **settings.quality_estimator_regressor.params
    )
    lc_regressor_cls = _get_class_from_str(
        settings.label_calibrator_regressor.regressor_class
    )
    lc_regressor_params = settings.label_calibrator_regressor.params

    if slc_settings:
        logger.info('Run training with Subthesauri Label Calibration')
        path_to_graph = slc_settings.thesaurus_file
        g = Graph()
        with timeit() as t:
            g.parse(str(path_to_graph))
        logger.debug('Parsed RDF Graph in %.4f seconds', t())

        if not slc_settings.subthesauri:
            logger.info(
                'No subthesauri passed, will extract subthesauri by type'
            )
        transformer = LabelCountForSubthesauriTransformer(
            graph=g,
            subthesaurus_type_uri=URIRef(slc_settings.subthesaurus_type),
            concept_type_uri=URIRef(slc_settings.concept_type),
            subthesauri=list(map(
                lambda s: URIRef(s), slc_settings.subthesauri
            )),
            concept_uri_prefix=slc_settings.concept_type_prefix
        )
        with timeit() as t:
            transformer.fit()
        logger.debug(
            'Ran Subthesauri Transformer fit in %.4f seconds', t()
        )
        label_calibration_features = ThesauriLabelCalibrationFeatures(
            transformer=transformer
        )
        features.append(label_calibration_features)
        t = Trainer(
            train_data=train_data,
            label_calibrator=ThesauriLabelCalibrator(
                regressor_class=lc_regressor_cls,
                regressor_params=lc_regressor_params,
                transformer=transformer),
            quality_regressor=quality_estimator,
            features=features,
            should_debug=settings.should_debug
        )
    else:
        logger.info('Run training with Simple Label Calibration')
        label_calibration_features = SimpleLabelCalibrationFeatures()
        features.append(label_calibration_features)
        t = Trainer(
            train_data=train_data,
            label_calibrator=SimpleLabelCalibrator(
                lc_regressor_cls(**lc_regressor_params)
            ),
            quality_regressor=quality_estimator,
            features=features,
            should_debug=settings.should_debug
        )

    model = t.train()
    logger.info('Store trained model in %s', path_to_model_output_file)
    dump(model, path_to_model_output_file)


def _get_class_from_str(fully_qualified_path: str) -> Type:
    split = fully_qualified_path.split('.')
    module = '.'.join(split[:-1])
    cls_name = split[-1]
    return getattr(import_module(module), cls_name)


def evaluate(settings: EvalSettings):
    logger = get_logger()
    path_to_test_tsv = settings.test_data_file
    path_to_model_file = settings.model_file
    model = load(path_to_model_file)
    logger.info('Run evaluation with model:\n%s', model)
    test_input = train_input_from_tsv(str(path_to_test_tsv))
    ev = Evaluator(test_input, model)
    eval_data = ev.evaluate()
    logger.info('\nScores:')
    for metric, score in eval_data.items():
        logger.info(f'{metric}: {score}')
