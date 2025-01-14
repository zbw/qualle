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
import re
from typing import Callable

import numpy as np

from qualle.features.base import Features


class TextCountFeature(Features):
    """Feature based on the occurence of something in a text."""

    def __init__(self, count_fun: Callable[[str], int]):
        self.count_fun = count_fun

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        ret = np.empty((len(X), 1))
        for i, x in enumerate(X):
            ret[i, 0] = self.count_fun(x)
        return ret


class CountCharsFeature(TextCountFeature):
    def __init__(self):
        super().__init__(self._count_char)

    @staticmethod
    def _count_char(txt: str):
        return len(txt)


class CountWordsFeature(TextCountFeature):
    def __init__(self):
        super().__init__(self._count_word)

    @staticmethod
    def _count_word(txt: str):
        return txt.count(" ")


class CountSpecialCharsFeature(TextCountFeature):
    _RE_SPECIAL = re.compile(r"""["'?!()]""")

    def __init__(self):
        super().__init__(self._count_special)

    @classmethod
    def _count_special(cls, txt: str):
        return len(cls._RE_SPECIAL.findall(txt))


class CountUpperCharsFeature(TextCountFeature):
    def __init__(self):
        super().__init__(self._count_upper)

    @staticmethod
    def _count_upper(txt: str):
        return sum((c.isupper() for c in txt))


class CountDigitsFeature(TextCountFeature):
    _RE_DIGIT = re.compile(r"\d")

    def __init__(self):
        super().__init__(self._count_digit)

    @classmethod
    def _count_digit(cls, txt: str):
        return len(cls._RE_DIGIT.findall(txt))
