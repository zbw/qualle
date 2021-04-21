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
