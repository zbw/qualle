#  Copyright 2021-2025 ZBW – Leibniz Information Centre for Economics
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
from pydantic import model_validator, BaseModel
from scipy.sparse import spmatrix

Labels = List[str]
Documents = List[str]
Scores = List[float]
Matrix = Union[np.ndarray, spmatrix]


class PredictData(BaseModel):
    docs: Documents
    predicted_labels: List[Labels]
    scores: List[Scores]

    @model_validator(mode="after")
    def check_equal_length(self):
        length = None
        for v in self.__dict__.values():
            if length is None:
                length = len(v)
            else:
                if length != len(v):
                    raise ValueError(
                        "docs, predicted_labels and scores "
                        "should have the same length"
                    )
        return self


class TrainData(BaseModel):

    predict_data: PredictData
    true_labels: List[Labels]

    @model_validator(mode="after")
    def check_equal_length(self):
        p_data = self.predict_data
        t_labels = self.true_labels
        if len(p_data.predicted_labels) != len(t_labels):
            raise ValueError("length of true labels and predicted labels do not match")
        return self


@dataclass
class LabelCalibrationData:

    predicted_no_of_labels: np.array
    predicted_labels: List[Labels]
