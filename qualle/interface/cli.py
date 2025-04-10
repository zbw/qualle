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

"""Parse command line args and call respective interface."""

import argparse
import json
import logging
import logging.config
from pathlib import Path

from qualle.interface.config import (
    TrainSettings,
    SubthesauriLabelCalibrationSettings,
    RegressorSettings,
    EvalSettings,
    FeaturesEnum,
    RESTSettings,
    PredictSettings,
)
from qualle.interface.internal import train, evaluate, predict
from qualle.interface.rest import run
from qualle.utils import get_logger

PATH_TO_MODEL_FILE_STR = "Path to model file"


def config_logging(config_file=None, debug=False):
    logger = get_logger()
    if config_file:
        logging.config.fileConfig(config_file)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)


class CliValidationError(Exception):
    pass


def handle_train(args: argparse.Namespace):
    logger = get_logger()

    if args.slc:
        if not all((args.thsys, args.s_type, args.c_type, args.c_uri_prefix)):
            raise CliValidationError(
                "Not all required arguments for Subthesauri Label Calibration "
                "have been passed. Please see usage."
            )
        if args.subthesauri:
            subthesauri = args.subthesauri[0].split(",")
        else:
            subthesauri = []

        slc = SubthesauriLabelCalibrationSettings(
            thesaurus_file=args.thsys[0],
            subthesaurus_type=args.s_type[0],
            concept_type=args.c_type[0],
            concept_type_prefix=args.c_uri_prefix[0],
            subthesauri=subthesauri,
            use_sparse_count_matrix=args.use_sparse_count_matrix,
        )
    else:
        slc = None

    features = []
    if args.features:
        cli_features = set(args.features)
        if {"all", "confidence"} & cli_features:
            features.append(FeaturesEnum.CONFIDENCE)
        if {"all", "text"} & cli_features:
            features.append(FeaturesEnum.TEXT)
        logger.debug(
            "Use features in addition to LabelCalibration: %s",
            list(map(lambda f: f.value, features)),
        )

    lc_regressor_config = json.loads(args.label_calibrator_regressor[0])
    lc_regressor = RegressorSettings(
        regressor_class=lc_regressor_config["class"],
        params={k: v for k, v in lc_regressor_config.items() if k != "class"},
    )
    qe_regressor_config = json.loads(args.quality_estimator_regressor[0])
    qe_regressor = RegressorSettings(
        regressor_class=qe_regressor_config["class"],
        params={k: v for k, v in qe_regressor_config.items() if k != "class"},
    )

    settings = TrainSettings(
        label_calibrator_regressor=lc_regressor,
        quality_estimator_regressor=qe_regressor,
        train_data_path=args.train_data_path,
        output_path=args.output,
        features=features,
        subthesauri_label_calibration=slc,
        should_debug=args.should_debug,
    )

    train(settings)


def handle_eval(args: argparse.Namespace):
    settings = EvalSettings(test_data_path=args.test_data_path, mdl_file=args.model)
    evaluate(settings)


def handle_rest(args: argparse.Namespace):
    settings = RESTSettings(mdl_file=args.model, port=args.port[0], host=args.host[0])
    run(settings)


def handle_predict(args: argparse.Namespace):
    predict_data_path = args.predict_data_path
    output_path = args.output[0] if args.output else None

    if predict_data_path.is_file() and not output_path:
        raise CliValidationError(
            "output file has to be specified if tsv file has been specified"
        )
    settings = PredictSettings(
        predict_data_path=predict_data_path,
        mdl_file=args.model,
        output_path=output_path,
    )
    predict(settings)


def add_eval_parser(subparsers: argparse._SubParsersAction):
    eval_parser = subparsers.add_parser(
        "eval", description="Run evaluation on training data."
    )
    eval_parser.set_defaults(func=handle_eval)

    eval_parser.add_argument(
        "test_data_path",
        type=str,
        help="Path to test data. "
        "Accepted inputs are either a tsv file or "
        "a folder in Annif format.",
    )
    eval_parser.add_argument("model", type=str, help=PATH_TO_MODEL_FILE_STR)


def add_slc_group(parsers: argparse.ArgumentParser):
    slc_group = parsers.add_argument_group(
        "subthesauri-label-calibration",
        description="Run Label Calibration by distinguishing between label "
        "count for different subthesauri. "
        "All Arguments in this group ,"
        " except --subthesauri , are required. "
        "If --subthesauri is not passed, subthesauri used "
        "are automatically detected by type.",
    )
    slc_group.add_argument(
        "--slc", action="store_true", help="Activate Subthesauri Label Calibration."
    )
    slc_group.add_argument(
        "--thsys",
        type=str,
        nargs=1,
        help="path to thesaurus file (must be in RDF format)",
    )
    slc_group.add_argument(
        "--s-type",
        type=str,
        nargs=1,
        help="subthesaurus type uri, e.g.:  "
        "http://zbw.eu/namespaces/zbw-extensions/Thsys",
    )
    slc_group.add_argument(
        "--c-type",
        type=str,
        nargs=1,
        help="concept type uri, e.g.: "
        "http://zbw.eu/namespaces/zbw-extensions/Descriptor",
    )
    slc_group.add_argument(
        "--c-uri-prefix",
        type=str,
        nargs=1,
        help="concept uri prefix, e.g.: http://zbw.eu/stw/descriptor)",
    )
    slc_group.add_argument(
        "--subthesauri",
        type=str,
        nargs=1,
        help="subthesauri URIs as comma-separated list, "
        "e.g.: http://zbw.eu/stw/thsys/v,"
        "http://zbw.eu/stw/thsys/b,"
        "http://zbw.eu/stw/thsys/n",
    )
    slc_group.add_argument(
        "--use-sparse-count-matrix",
        action="store_true",
        help="Use matrix in SciPy Sparse format to store count of labels per"
        " Subthesauri in memory. Useful to reduce memory usage when"
        " mapping labels to subthesauri "
        " (used for label calibration training and"
        "  for calculating the input of the recall predictor)"
        " if the distribution of labels over Subthesauri is sparse.",
    )


def add_train_parser(subparsers: argparse._SubParsersAction):
    train_parser = subparsers.add_parser(
        "train", description="Run training using default estimator."
    )
    train_parser.set_defaults(func=handle_train)
    train_parser.add_argument(
        "train_data_path",
        type=str,
        help="Path to train data. "
        "Accepted inputs are either a tsv file or "
        "a folder in Annif format.",
    )
    train_parser.add_argument("output", type=str, help="Path to output model file")
    train_parser.add_argument(
        "--features",
        "-f",
        choices=["all", "confidence", "text"],
        action="append",
        type=str,
        help="Use features in addition to Label Calibration. "
        "Can be passed multiple times. "
        'If "all" is passed, all features will be used.',
    )
    train_parser.add_argument(
        "--label-calibrator-regressor",
        type=str,
        nargs=1,
        help="Specifiy regressor to use for Label Calibration in JSON format."
        'Requires property "class" with fully qualified name of the '
        "scikit-learn regressor class to use  for the Label Calibrator. "
        'E.g.: {"class": "sklearn.ensemble.GradientBoostingRegressor",'
        '"min_samples_leaf": 30, "max_depth": 5, "n_estimators": 10} ',
        default=[
            '{"class": "sklearn.ensemble.GradientBoostingRegressor",'
            '"min_samples_leaf": 30, "max_depth": 5, "n_estimators": 10}'
        ],
    )
    train_parser.add_argument(
        "--quality-estimator-regressor",
        type=str,
        nargs=1,
        help="Specifiy regressor to use for Label Calibration in JSON format."
        'Requires property "class" with fully qualified name of the '
        "scikit-learn regressor class to use  for the Quality Estimator. "
        'E.g.: {"class": "sklearn.ensemble.GradientBoostingRegressor",'
        '"min_samples_leaf": 30, "max_depth": 5, "n_estimators": 10} ',
        default=[
            '{"class": "sklearn.ensemble.GradientBoostingRegressor",'
            '"n_estimators": 10, "max_depth": 8}'
        ],
    )
    add_slc_group(train_parser)


def add_rest_parser(subparsers: argparse._SubParsersAction):
    rest_parser = subparsers.add_parser(
        "rest", description="Run Qualle as REST Webservice with predict endpoint"
    )
    rest_parser.add_argument("model", type=str, help=PATH_TO_MODEL_FILE_STR)
    rest_parser.add_argument(
        "--port", "-p", type=int, nargs=1, help="Port to listen on", default=[8000]
    )
    rest_parser.add_argument(
        "--host", type=str, nargs=1, help="Host to listen on", default=["127.0.0.1"]
    )
    rest_parser.set_defaults(func=handle_rest)


def add_predict_parser(subparsers: argparse._SubParsersAction):
    predict_parser = subparsers.add_parser(
        "predict",
        description="Predict the quality for a collection of automated "
        "subject indexing (MLC) results.",
    )
    predict_parser.set_defaults(func=handle_predict)

    predict_parser.add_argument(
        "predict_data_path",
        type=Path,
        help="Path to data. "
        "Accepted inputs are either a tsv file or a folder in annif "
        "format. If a tsv file is provided, the flag --output must be "
        "specified. If a folder is provided, the predicted quality is "
        'written into files using the ".qualle" prefix inside the folder.',
    )
    predict_parser.add_argument("model", type=str, help=PATH_TO_MODEL_FILE_STR)
    predict_parser.add_argument(
        "--output",
        type=Path,
        nargs=1,
        help="Output file to write the quality estimation. Only used if a tsv "
        "file is specified as input. The quality estimation for each "
        "line of the tsv file is written to the output on a separate "
        "line, preserving the order in the tsv file.",
    )


def cli_entrypoint():
    parser = argparse.ArgumentParser(
        description="Quality Estimation for Automatic Subject Indexing"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set Log Level to Debug",
        dest="should_debug",
    )
    parser.add_argument(
        "--logging-conf",
        nargs=1,
        type=str,
        help="Path to logging config file in configparser format. "
        'The name of the logger which has to be configured is "qualle"',
    )
    subparsers = parser.add_subparsers(
        title="Subcommands", required=True, dest="command"
    )

    add_eval_parser(subparsers)
    add_train_parser(subparsers)
    add_rest_parser(subparsers)
    add_predict_parser(subparsers)

    args = parser.parse_args()

    config_logging(config_file=args.logging_conf, debug=args.should_debug)
    args.func(args)
