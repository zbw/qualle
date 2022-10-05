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
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from scipy.sparse import spmatrix

Labels = List[str]
Documents = List[str]
Scores = List[float]
Matrix = Union[np.ndarray, spmatrix]


@dataclass
class PredictData:
    docs: Documents
    predicted_labels: List[Labels]
    scores: List[Scores]


@dataclass
class TrainData:
    predict_data: PredictData
    true_labels: List[Labels]


@dataclass
class LabelCalibrationData:

    predicted_no_of_labels: np.array
    predicted_labels: List[Labels]
