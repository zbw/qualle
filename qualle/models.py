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
from dataclasses import dataclass
from typing import List

import numpy as np

Labels = List[str]
Documents = List[str]


@dataclass
class TrainData:

    docs: Documents
    predicted_labels: List[Labels]
    true_labels: List[Labels]


@dataclass
class PredictData:

    docs: Documents
    predicted_labels: List[Labels]


@dataclass
class LabelCalibrationData:

    predicted_no_of_labels: np.array
    predicted_labels: List[Labels]
