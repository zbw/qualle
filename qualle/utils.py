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
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import List

from qualle.models import Labels


def recall(true_labels: List[Labels], predicted_labels: List[Labels]) -> List:
    return [
        len(set(tc) & set(pc)) / len_tc if (len_tc := len(tc)) > 0 else 0
        for tc, pc in zip(true_labels, predicted_labels)
    ]


def get_logger():
    return logging.getLogger("qualle")


@contextmanager
def timeit():
    """Context manager to time the code block wrapped by the manager.

    Yielded object is a lambda function which can be called to compute the
    duration (in seconds) of the executed code block.
    """
    start = perf_counter()
    yield lambda: perf_counter() - start
