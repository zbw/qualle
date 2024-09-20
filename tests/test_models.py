#  Copyright 2021-2023 ZBW – Leibniz Information Centre for Economics
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
import pytest
from pydantic import ValidationError

from qualle.models import PredictData, TrainData


DUMMY_DOCS = ["doc1", "doc2"]
DUMMY_PRED_LABELS_1 = [["label 1"], ]
DUMMY_PRED_LABELS_2 = [["label 1"], ["label 2"], ]
DUMMY_SCORES_1 = [[2.5, 3.0], ]
DUMMY_SCORES_2 = [[2.5, 3.0], [5.5, 7.2], ]
DUMMY_SCORES_3 = [[2.5, 3.0], [5.5, 7.2], [11.5, 23.1]]
DUMMY_TRUE_LABELS = [["true label 1"], ["true label 2"], ]


def test_unequal_length_in_predict_data_raises_validator_exc_1():
    with pytest.raises(ValidationError):
        PredictData(
            docs=DUMMY_DOCS,
            predicted_labels=DUMMY_PRED_LABELS_1,
            scores=DUMMY_SCORES_1,
        )


def test_unequal_length_in_predict_data_raises_validator_exc_2():
    with pytest.raises(ValidationError):
        PredictData(
            docs=DUMMY_DOCS,
            predicted_labels=DUMMY_PRED_LABELS_1,
            scores=DUMMY_SCORES_3,
        )


def test_missing_attrbute_in_predict_data_raises_validator_exc():
    with pytest.raises(ValidationError):
        PredictData(
            docs=DUMMY_DOCS,
            predicted_labels=DUMMY_PRED_LABELS_1,
        )


def test_predict_data_no_validation_exc():
    PredictData(
        docs=DUMMY_DOCS,
        predicted_labels=DUMMY_PRED_LABELS_2,
        scores=DUMMY_SCORES_2,
    )


def test_unequal_length_in_train_data_raises_validator_exc_():
    with pytest.raises(ValidationError):
        TrainData(
            predict_data=PredictData(
                docs=DUMMY_DOCS,
                predicted_labels=DUMMY_PRED_LABELS_2,
                scores=DUMMY_SCORES_2,
            ),
            true_labels=DUMMY_PRED_LABELS_1,
        )


def test_missing_attribute_in_train_data_raises_validator_exc_():
    with pytest.raises(ValidationError):
        TrainData(
            predict_data=PredictData(
                docs=DUMMY_DOCS,
                predicted_labels=DUMMY_PRED_LABELS_2,
                scores=DUMMY_SCORES_2,
            ),
        )


def test_train_data_no_validator_exc_():
    TrainData(
        predict_data=PredictData(
            docs=DUMMY_DOCS,
            predicted_labels=DUMMY_PRED_LABELS_2,
            scores=DUMMY_SCORES_2,
        ),
        true_labels=DUMMY_TRUE_LABELS
    )
