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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion

from qualle.features.text.count import (
    CountCharsFeature,
    CountWordsFeature,
    CountSpecialCharsFeature,
    CountUpperCharsFeature,
    CountDigitsFeature,
)

_NAME_VECTOR_FEATURE = "vectorizer"
_NAME_CHAR_FEATURE = "n_chars"
_NAME_WORD_FEATURE = "n_words"
_NAME_SPECIAL_CHARS_FEATURE = "n_special"
_NAME_UPPER_FEATURE = "n_upper"
_NAME_DIGIT_FEATURE = "n_digits"


class TextFeatures(FeatureUnion):
    """Features based on the text of a document"""

    def __init__(self):
        super().__init__(
            [
                (_NAME_VECTOR_FEATURE, CountVectorizer(lowercase=False)),
                (_NAME_CHAR_FEATURE, CountCharsFeature()),
                (_NAME_WORD_FEATURE, CountWordsFeature()),
                (_NAME_SPECIAL_CHARS_FEATURE, CountSpecialCharsFeature()),
                (_NAME_UPPER_FEATURE, CountUpperCharsFeature()),
                (_NAME_DIGIT_FEATURE, CountDigitsFeature()),
            ]
        )
