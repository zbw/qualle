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
from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np
from pydantic import root_validator, BaseModel
from scipy.sparse import spmatrix

Label = str
Labels = List[Label]
Document = str
Documents = List[Document]
Score = float
Scores = List[Score]
Matrix = Union[np.ndarray, spmatrix]


class PredictData(BaseModel):
    docs: Documents
    predicted_labels: List[Labels]
    scores: List[Scores]

    @root_validator
    def check_equal_length(cls, values):
        length = None
        for v in values.values():
            if length is None:
                length = len(v)
            else:
                if length != len(v):
                    raise ValueError(
                        "docs, predicted_labels and scores "
                        "should have the same length"
                    )
        return values


class LabelCalibrationTrainData(BaseModel):
    docs: Documents
    true_labels: List[Labels]

    @root_validator
    def check_equal_length(cls, values):
        docs = values.get("docs")
        t_labels = values.get("true_labels")
        if len(docs) != len(t_labels):
            raise ValueError("length of true labels and docs do not match")
        return values


class PredictTrainData(BaseModel):
    predict_data: PredictData
    true_labels: List[Labels]

    @root_validator
    def check_equal_length(cls, values):
        p_data = values.get("predict_data")
        t_labels = values.get("true_labels")
        if len(p_data.predicted_labels) != len(t_labels):
            raise ValueError("length of true labels and predicted labels do not match")
        return values


EvalData = PredictTrainData


class TrainData(BaseModel):
    label_calibration_split: Optional[LabelCalibrationTrainData]
    predict_split: PredictTrainData


@dataclass
class LabelCalibrationData:

    predicted_no_of_labels: np.array
    predicted_labels: List[Labels]
