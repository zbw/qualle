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
from typing import List

from sklearn.utils.validation import check_is_fitted
from stwfsapy.text_features import mk_text_features

from qualle.features.base import Features


class TextFeatures(Features):
    """Features based on the text string of a document."""

    def fit(self, X: List[str], y=None):
        self.features_ = mk_text_features()
        self.features_.fit(X)
        return self

    def transform(self, X: List[str]):
        check_is_fitted(self)
        return self.features_.transform(X)
