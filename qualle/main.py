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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Quality Estimation for Automatic Subject Indexing'
    )

    parser.add_argument(
        '--debug', action='store_true', help='Set Log Level to Debug',
        dest='should_debug'
    )
    parser.add_argument(
        '--eval', '-e', type=str, nargs=2,
        help='Run evaluation on training data using different estimators. '
             'Specify test tsv and model input file as arguments.'
             ' E.g.: --eval test.tsv input.model',
        metavar=('test_input', 'model_input')
    )
    parser.add_argument(
        '--train', '-t', type=str, nargs=2,
        help='Run training using default estimator.\n'
        'Specify train input and model output file as arguments.'
        ' E.g.: --train train.tsv output.model',
        metavar=('train_file', 'output_file')
    )
    parser.add_argument(
        '--subthesauri-label-calibration', type=str, nargs=6,
        help='Run Label Calibration by distinguishing between label count for'
             ' different subthesauri. Only valid in conjunction with '
             '--train option. Requires following arguments:'
             '\t- path to thesaurus file (must be in RDF format)'
             '\t- subthesaurus type uri, e.g.:  '
             'http://zbw.eu/namespaces/zbw-extensions/Thsys'
             '\t- concept type uri, e.g.: '
             'http://zbw.eu/namespaces/zbw-extensions/Descriptor'
             '\t- concept uri prefix, e.g.: http://zbw.eu/stw/descriptor'
             '\t- subthesaurus uri prefix, e.g.: http://zbw.eu/stw/thsys'
             '\t- subthesauri ids as comma-separated list, '
             'e.g.: v,b,n,w,p,g,a',
        metavar=(
            'thesaurus', 'subthesaurus_type_uri', 'concept_type_uri',
            'concept_uri_prefix', 'subthesaurus_uri_prefix', 'subthesauri_ids'
        )
    )

    args = parser.parse_args()

    logger = config_logging(args.should_debug)

    if args.eval:
        path_to_test_tsv = args.eval[0]
        path_to_model_file = args.eval[1]
        model = load(path_to_model_file)
        logger.info('Run evaluation with model:\n%s', model)
        test_input = train_input_from_tsv(path_to_test_tsv)
        ev = Evaluator(test_input, model)
        eval_data = ev.evaluate()
        logger.info('\nScores:')
        for metric, score in eval_data.items():
            logger.info(f'{metric}: {score}')
    elif args.train:
        path_to_train_tsv = args.train[0]
        path_to_model_output_file = args.train[1]
        train_data = train_input_from_tsv(path_to_train_tsv)

        if slc_args := args.subthesauri_label_calibration:
            logger.info('Run training with Subthesauri Label Calibration')

            path_to_graph = slc_args[0]
            g = Graph()
            with timeit() as t:
                g.parse(path_to_graph)
            logger.debug('Parsed RDF Graph in %.4f seconds', t())

            subthesaurus_prefix = slc_args[4]
            subthesauri = slc_args[5].split(',')
            transformer = LabelCountForSubthesauriTransformer(
                graph=g,
                subthesaurus_type_uri=URIRef(slc_args[1]),
                concept_type_uri=URIRef(slc_args[2]),
                subthesauri=list(map(
                    lambda s: URIRef(
                        subthesaurus_prefix.rstrip('/') + '/' + s),
                    subthesauri
                )),
                concept_uri_prefix=slc_args[3]
            )
            with timeit() as t:
                transformer.fit()
            logger.debug(
                'Ran Subthesauri Transformer fit in %.4f seconds', t()
            )
            t = Trainer(
                train_data=train_data,
                label_calibrator=ThesauriLabelCalibrator(
                    regressor_class=GradientBoostingRegressor,
                    regressor_params=dict(
                        min_samples_leaf=30, max_depth=5, n_estimators=10),
                    transformer=transformer),
                label_calibration_features=ThesauriLabelCalibrationFeatures(
                    transformer=transformer
                ),
                should_debug=args.should_debug
            )
        else:
            logger.info('Run training with Simple Label Calibration')
            t = Trainer(
                train_data=train_data,
                label_calibrator=SimpleLabelCalibrator(
                    GradientBoostingRegressor(
                        min_samples_leaf=30, max_depth=5, n_estimators=10)
                ),
                label_calibration_features=SimpleLabelCalibrationFeatures(),
                should_debug=args.should_debug
            )

        model = t.train()
        logger.info('Store trained model in %s', path_to_model_output_file)
        dump(model, path_to_model_output_file)
