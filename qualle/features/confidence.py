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
from typing import List

import numpy as np

from qualle.features.base import Features
from qualle.models import Scores


class ConfidenceFeatures(Features):
    """Features based on aggregating concept level confidence scores."""

    def transform(self, X: List[Scores]):
        _min = [np.min(row) if row else 0.0 for row in X]
        _mean = [np.mean(row) if row else 0.0 for row in X]
        _median = [np.median(row) if row else 0.0 for row in X]
        _prod = [np.prod(row) if row else 0.0 for row in X]
        return np.column_stack([_min, _mean, _median, _prod])
