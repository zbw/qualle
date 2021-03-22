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
from typing import List

from qualle.models import Concepts, TrainData


def recall(
        true_concepts: List[Concepts], predicted_concepts: List[Concepts]
) -> List:
    return [len(set(tc) & set(pc)) / len(tc) for tc, pc in zip(
        true_concepts, predicted_concepts)
     ]


def train_input_from_tsv(
        path_to_tsv: str
) -> TrainData:
    docs = []
    pred_concepts = []
    true_concepts = []

    with open(path_to_tsv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            docs.append(row[0])
            pred_concepts.append(list(
                map(lambda s: s.split(':')[0], row[1].split(','))
            ))
            true_concepts.append(row[2].split(','))

    return TrainData(
        docs=docs, predicted_concepts=pred_concepts,
        true_concepts=true_concepts
    )


def get_logger():
    return logging.getLogger('qualle')
