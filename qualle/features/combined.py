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
from typing import List, Dict, Any, Type

import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_is_fitted

from qualle.features.base import Features


CombinedFeaturesData = Dict[Type[Features], Any]


class CombinedFeatures(Features):
    """Combine n features by horizontal stacking"""

    def __init__(self, features: List[Features]):
        self.features = features

    def fit(self, X: CombinedFeaturesData, y=None):
        self.features_ = self.features
        for f in self.features_:
            f.fit(X[f.__class__])
        return self

    def transform(self, X: CombinedFeaturesData):
        check_is_fitted(self)
        combined = [f.transform(X[f.__class__]) for f in self.features_]
        issparse = any(map(sp.issparse, combined))
        if issparse:
            return sp.hstack(combined, format="csr")
        else:
            return np.hstack(combined)
