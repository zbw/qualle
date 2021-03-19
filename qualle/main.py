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

from qualle.evaluate import Evaluator
from qualle.utils import write_to_tsv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Content-Based Quality Estimation for Automatic Subject '
                    'Indexing of Short Texts under Precision and Recall '
                    'Constraints'
    )

    parser.add_argument(
        '--eval', '-e', type=str, nargs=3,
        help='Run evaluation on training data using different estimators. '
             'Specify train and eval tsv and output directoy as arguments.'
             ' E.g.: --eval train.tsv eval.tsv output_dir')

    args = parser.parse_args()

    if args.eval:
        path_to_train_tsv = args.eval[0]
        path_to_eval_tsv = args.eval[1]
        path_to_out_dir = args.eval[2]
        ev = Evaluator(path_to_train_tsv, path_to_eval_tsv)
        eval_data = ev.evaluate()
        write_to_tsv(
            eval_data, path_to_out_dir.rstrip('/') + '/' +
            'qe_avg_metrics.tsv'
        )
