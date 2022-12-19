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

import pytest

from qualle.features.text import CountCharsFeature, CountWordsFeature, \
    CountSpecialCharsFeature, CountUpperCharsFeature, CountDigitsFeature
from qualle.features.text.count import TextCountFeature


@pytest.fixture()
def text():
    return (
        "abcdefghijklmnopqrstuvwxyzäöü"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"
        " "
        "0123456789"
        "\"'?!()&%$"
    )


def test_count_char(text):
    assert CountCharsFeature()._count_char(text) == 78


def test_count_words(text):
    assert CountWordsFeature()._count_word(text) == 1


def test_count_special(text):
    assert CountSpecialCharsFeature()._count_special(text) == 6


def test_count_upper(text):
    assert CountUpperCharsFeature()._count_upper(text) == 29


def test_count_digit(text):
    assert CountDigitsFeature()._count_digit(text) == 10


def test_transform():
    c_feature = TextCountFeature(
        lambda d: sum(map(lambda s: int(s) if s not in "abc" else 0, d))
    )
    c_feature.fit([])
    transformed = c_feature.transform(["0123456789abc"[:i] for i in range(14)])
    assert len(transformed) == 14
    for i in range(10):
        assert list(transformed[i]) == [sum(range(i))]
    for i in range(10, 14):
        assert list(transformed[i]) == [sum(range(10))]
