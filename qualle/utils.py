#  Copyright 2021-2022 ZBW – Leibniz Information Centre for Economics
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
import csv
import logging
from contextlib import contextmanager
from time import perf_counter
from glob import glob
import os
from typing import List


from qualle.models import Labels, TrainData, PredictData


def recall(
        true_labels: List[Labels], predicted_labels: List[Labels]
) -> List:
    return [
        len(set(tc) & set(pc)) / len_tc if (len_tc := len(tc)) > 0 else 0
        for tc, pc in zip(true_labels, predicted_labels)
    ]


def load_train_input(pth_to_data: str) -> TrainData:
    if os.path.isdir(pth_to_data):
        return train_input_from_annif(pth_to_data)
    else:
        return train_input_from_tsv(pth_to_data)


def train_input_from_tsv(
        path_to_tsv: str
) -> TrainData:
    docs = []
    pred_labels = []
    true_labels = []
    scores = []

    with open(path_to_tsv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            docs.append(row[0])
            pred_labels_for_row = []
            scores_for_row = []
            label_score_pairs = row[1].split(',')
            if len(label_score_pairs) > 0 and label_score_pairs[0]:
                for label_score_pair in label_score_pairs:
                    label, score = label_score_pair.split(':')
                    pred_labels_for_row.append(label)
                    scores_for_row.append(float(score))
            pred_labels.append(pred_labels_for_row)
            scores.append(scores_for_row)
            true_labels.append(list(filter(bool, row[2].split(','))))

    return TrainData(
        PredictData(docs=docs, predicted_labels=pred_labels, scores=scores),
        true_labels=true_labels
    )


def train_input_from_annif(path_to_folder: str) -> TrainData:
    docs = []
    pred_labels = []
    true_labels = []
    scores = []
    for pred_pth in glob((os.path.join(path_to_folder, '*.annif'))):
        with open(pred_pth) as fp:
            scores_for_doc = []
            pred_labels_for_doc = []
            for line in fp.readlines():
                split = line.rstrip('\n').split('\t')
                concept_id = extract_concept_id_from_annif_label(split[0])
                pred_labels_for_doc.append(concept_id)
                score = float(split[2])
                scores_for_doc.append(score)
            pred_labels.append(pred_labels_for_doc)
            scores.append(scores_for_doc)
        doc_pth = pred_pth.replace('.annif', '.txt')
        with open(doc_pth) as fp:
            docs.append(''.join(list(fp.readlines())))
        label_pth = pred_pth.replace('.annif', '.tsv')
        if os.path.exists(label_pth):
            with open(label_pth) as fp:
                true_labels_for_doc = []
                for line in fp.readlines():
                    split = line.rstrip('\n').split('\t')
                    concept_id = extract_concept_id_from_annif_label(split[0])
                    true_labels_for_doc.append(concept_id)
                true_labels.append(true_labels_for_doc)
    return TrainData(
        PredictData(docs=docs, predicted_labels=pred_labels, scores=scores),
        true_labels=true_labels
    )


def extract_concept_id_from_annif_label(uri):
    split = uri.split('/')
    # remove '>' at end of URI
    return split[-1][:-1]


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
