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


import argparse
import logging

from joblib import dump, load

from rdflib import URIRef, Graph
from sklearn.ensemble import GradientBoostingRegressor

from qualle.evaluate import Evaluator
from qualle.features.confidence import ConfidenceFeatures
from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrator, SimpleLabelCalibrationFeatures
from qualle.features.label_calibration.thesauri_label_calibration import \
    ThesauriLabelCalibrator, ThesauriLabelCalibrationFeatures, \
    LabelCountForSubthesauriTransformer
from qualle.train import Trainer
from qualle.utils import get_logger, train_input_from_tsv, timeit


def config_logging(should_debug):
    logger = get_logger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if should_debug else logging.INFO)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG if should_debug else logging.INFO)
    return logger


def handle_eval(args):
    logger = get_logger()
    path_to_test_tsv = args.test_data_file
    path_to_model_file = args.model
    model = load(path_to_model_file)
    logger.info('Run evaluation with model:\n%s', model)
    test_input = train_input_from_tsv(path_to_test_tsv)
    ev = Evaluator(test_input, model)
    eval_data = ev.evaluate()
    logger.info('\nScores:')
    for metric, score in eval_data.items():
        logger.info(f'{metric}: {score}')


def handle_train(args):
    logger = get_logger()
    path_to_train_tsv = args.train_data_file
    path_to_model_output_file = args.output
    train_data = train_input_from_tsv(path_to_train_tsv)

    features = []
    if args.confidence:
        features.append(ConfidenceFeatures())

    if args.slc:
        if not all((args.thsys, args.s_type, args.c_type, args.c_uri_prefix)):
            raise Exception(
                'Not all required arguments for Subthesauri Label Calibration '
                'have been passed. Please see usage.'
            )

        logger.info('Run training with Subthesauri Label Calibration')
        path_to_graph = args.thsys[0]
        g = Graph()
        with timeit() as t:
            g.parse(path_to_graph)
        logger.debug('Parsed RDF Graph in %.4f seconds', t())

        if args.subthesauri:
            subthesauri = list(map(
                lambda s: URIRef(s), args.subthesauri[0].split(',')
            ))
        else:
            subthesauri = None
            logger.info(
                'No subthesauri passed, will extract subthesauri by type'
            )
        transformer = LabelCountForSubthesauriTransformer(
            graph=g,
            subthesaurus_type_uri=URIRef(args.s_type[0]),
            concept_type_uri=URIRef(args.c_type[0]),
            subthesauri=subthesauri,
            concept_uri_prefix=args.c_uri_prefix[0]
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
                regressor_class=GradientBoostingRegressor,
                regressor_params=dict(
                    min_samples_leaf=30, max_depth=5, n_estimators=10),
                transformer=transformer),
            recall_predictor_regressor=GradientBoostingRegressor(
                n_estimators=10, max_depth=8),
            features=features,
            should_debug=args.should_debug
        )
    else:
        logger.info('Run training with Simple Label Calibration')
        label_calibration_features = SimpleLabelCalibrationFeatures()
        features.append(label_calibration_features)
        t = Trainer(
            train_data=train_data,
            label_calibrator=SimpleLabelCalibrator(
                GradientBoostingRegressor(
                    min_samples_leaf=30, max_depth=5, n_estimators=10)
            ),
            recall_predictor_regressor=GradientBoostingRegressor(
                n_estimators=10, max_depth=8),
            features=features,
            should_debug=args.should_debug
        )

    model = t.train()
    logger.info('Store trained model in %s', path_to_model_output_file)
    dump(model, path_to_model_output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quality Estimation for Automatic Subject Indexing'
    )

    parser.add_argument(
        '--debug', action='store_true', help='Set Log Level to Debug',
        dest='should_debug'
    )
    subparsers = parser.add_subparsers(title='Subcommands')

    eval_parser = subparsers.add_parser(
        'eval',
        description='Run evaluation on training data using different '
                    'estimators.'
    )
    eval_parser.set_defaults(func=handle_eval)

    eval_parser.add_argument('test_data_file', type=str,
                             help='Path to test data file in tsv format')
    eval_parser.add_argument('model', type=str, help='Path to model file')

    train_parser = subparsers.add_parser(
        'train',
        description='Run training using default estimator.'
    )
    train_parser.set_defaults(func=handle_train)
    train_parser.add_argument('train_data_file', type=str,
                              help='Path to train data file in tsv format')
    train_parser.add_argument('output', type=str,
                              help='Path to output model file')
    train_parser.add_argument('--confidence', action='store_true',
                              help='Use confidence features')
    slc_group = train_parser.add_argument_group(
        'subthesauri-label-calibration',
        description='Run Label Calibration by distinguishing between label '
                    'count for different subthesauri. '
                    'All Arguments in this group ,'
                    ' except --subthesauri , are required. '
                    'If --subthesauri is not passed, subthesauri used '
                    'are automatically detected by type.')
    slc_group.add_argument('--slc', action='store_true',
                           help='Activate Subthesauri Label Calibration.')
    slc_group.add_argument(
        '--thsys', type=str, nargs=1,
        help='path to thesaurus file (must be in RDF format)')
    slc_group.add_argument(
        '--s-type', type=str, nargs=1,
        help='subthesaurus type uri, e.g.:  '
             'http://zbw.eu/namespaces/zbw-extensions/Thsys')
    slc_group.add_argument(
        '--c-type', type=str, nargs=1,
        help='concept type uri, e.g.: '
             'http://zbw.eu/namespaces/zbw-extensions/Descriptor')
    slc_group.add_argument(
        '--c-uri-prefix', type=str, nargs=1,
        help='concept uri prefix, e.g.: http://zbw.eu/stw/descriptor)')
    slc_group.add_argument('--subthesauri', type=str, nargs=1,
                           help='subthesauri URIs as comma-separated list, '
                                'e.g.: http://zbw.eu/stw/thsys/v,'
                                'http://zbw.eu/stw/thsys/b,'
                                'http://zbw.eu/stw/thsys/n')

    args = parser.parse_args()

    config_logging(args.should_debug)
    args.func(args)
