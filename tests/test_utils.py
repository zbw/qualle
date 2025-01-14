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

from qualle.utils import recall, timeit


def test_recall():
    true_labels = [["x0", "x2"], ["x1"], ["x3"]]
    pred_labels = [["x0", "x1"], ["x2"], ["x3"]]

    assert recall(true_labels=true_labels, predicted_labels=pred_labels) == [0.5, 0, 1]


def test_recall_empty_true_labels_return_zero():
    assert recall(true_labels=[[]], predicted_labels=[["x"]]) == [0]


def test_recall_empty_pred_labels_return_zero():
    assert recall(true_labels=[["x"]], predicted_labels=[[]]) == [0]


def test_recall_empty_input():
    assert recall(true_labels=[], predicted_labels=[]) == []


def test_timeit(mocker):
    m = mocker.Mock(side_effect=[1, 3])
    mocker.patch("qualle.utils.perf_counter", m)
    with timeit() as t:
        _ = 1 + 1
    assert t() == 2
