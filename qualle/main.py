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

from qualle.evaluate import Evaluator
from qualle.train import Trainer
from qualle.utils import get_logger, train_input_from_tsv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Quality Estimation for Automatic Subject Indexing'
    )

    parser.add_argument(
        '--eval', '-e', type=str, nargs=2,
        help='Run evaluation on training data using different estimators. '
             'Specify test tsv and model input file as arguments.'
             ' E.g.: --eval test.tsv input.model'
    )
    parser.add_argument(
        '--train', '-t', type=str, nargs=2,
        help='Run training using default estimator.\n'
        'Specify train input and model output file as arguments.'
        ' E.g.: --train train.tsv output.model'
    )

    args = parser.parse_args()

    if args.eval:
        path_to_test_tsv = args.eval[0]
        path_to_model_file = args.eval[1]
        model = load(path_to_model_file)
        test_input = train_input_from_tsv(path_to_test_tsv)
        ev = Evaluator(test_input, model)
        eval_data = ev.evaluate()
        logger = get_logger()
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        for metric, score in eval_data.items():
            logger.info(f'{metric}: {score}')
    elif args.train:
        path_to_train_tsv = args.train[0]
        path_to_model_output_file = args.train[1]
        train_data = train_input_from_tsv(path_to_train_tsv)

        t = Trainer(train_data)

        model = t.train()
        dump(model, path_to_model_output_file)
