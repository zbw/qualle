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
import csv
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import List


from qualle.models import Labels, TrainData, PredictData


def recall(
        true_labels: List[Labels], predicted_labels: List[Labels]
) -> List:
    return [
        len(set(tc) & set(pc)) / len_tc if (len_tc := len(tc)) > 0 else 0
        for tc, pc in zip(true_labels, predicted_labels)
    ]


def train_input_from_tsv(
        path_to_tsv: str
) -> TrainData:
    docs = []
    pred_labels = []
    true_labels = []

    with open(path_to_tsv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            docs.append(row[0])
            pred_labels.append(list(
                    filter(bool, map(
                        lambda s: s.split(':')[0], row[1].split(',')
                    ))
            ))
            true_labels.append(list(filter(bool, row[2].split(','))))

    return TrainData(
        PredictData(docs=docs, predicted_labels=pred_labels),
        true_labels=true_labels
    )


def get_logger():
    return logging.getLogger('qualle')


@contextmanager
def timeit():
    """Context manager to time the code block wrapped by the manager.

    Yielded object is a lambda function which can be called to compute the
    duration (in seconds) of the executed code block.
    """
    start = perf_counter()
    yield lambda: perf_counter() - start
