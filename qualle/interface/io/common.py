#  Copyright 2021-2023 ZBW  – Leibniz Information Centre for Economics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from collections import namedtuple

from qualle.models import TrainData, PredictData

Data = namedtuple(
    'data', ['docs', 'predicted_labels', 'scores', 'true_labels']
)


def map_to_train_data(data: Data) -> TrainData:
    return TrainData(
        predict_data=map_to_predict_data(data),
        true_labels=data.true_labels
    )


def map_to_predict_data(data: Data) -> PredictData:
    return PredictData(
        docs=data.docs,
        predicted_labels=data.predicted_labels,
        scores=data.scores
    )
