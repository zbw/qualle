#  Copyright 2021-2023 ZBW  – Leibniz Information Centre for Economics
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import pytest

from qualle.interface.config import EvalSettings

from qualle.interface import evaluation as e
from qualle.models import PredictData, EvalData
from tests.interface.common import URI_PREFIX


@pytest.fixture(autouse=True)
def mock_io(mocker):
    mocker.patch("qualle.interface.evaluation.load_model")


@pytest.fixture
def eval_data():
    p = PredictData(
        docs=[f"Title{i}" for i in range(20)],
        predicted_labels=[[f"concept{i}"] for i in range(20)],
        scores=[[i / 20] for i in range(20)],
    )
    return EvalData(predict_data=p, true_labels=[[f"concept{i}"] for i in range(20)])


@pytest.fixture
def tsv_evaluate_data_path(tmp_path):
    tsv_file = tmp_path / "evaluate-docs.tsv"
    content = "\n".join(f"Title{i}\tconcept{i}:{i / 20}\tconcept{i}" for i in range(20))
    tsv_file.write_text(content)
    return tsv_file


@pytest.fixture
def annif_evaluate_data_dir(tmp_path):
    p_dir = tmp_path / "predict"
    p_dir.mkdir()
    for i in range(20):
        doc0 = p_dir / f"doc{i}.txt"
        doc0.write_text(f"Title{i}")
        scores0 = p_dir / f"doc{i}.annif"
        scores0.write_text(f"<{URI_PREFIX}concept{i}>\tlabel0\t{i / 20}\n")

        labels0 = p_dir / f"doc{i}.tsv"
        labels0.write_text(f"<{URI_PREFIX}concept{i}>\tlabel1\n")
    return p_dir


def test_evaluate(mocker, tsv_evaluate_data_path, eval_data, model_path):
    m_eval = mocker.Mock()
    m_eval.evaluate.return_value = {}
    m_eval_cls = mocker.Mock(return_value=m_eval)
    mocker.patch("qualle.interface.evaluation.Evaluator", m_eval_cls)
    e.load_model.return_value = "testmodel"

    settings = EvalSettings(
        test_data_path=tsv_evaluate_data_path, model_file=model_path
    )
    e.evaluate(settings)

    m_eval_cls.assert_called_once_with(eval_data, "testmodel")
    m_eval.evaluate.assert_called_once()


def test_load_eval_input_from_annif(annif_evaluate_data_dir, eval_data):
    actual_eval_data = e._load_eval_input(annif_evaluate_data_dir)

    actual_eval_data_tpls = zip(
        actual_eval_data.predict_data.docs,
        actual_eval_data.predict_data.predicted_labels,
        actual_eval_data.predict_data.scores,
        actual_eval_data.true_labels,
    )
    expected_eval_data_tpls = zip(
        eval_data.predict_data.docs,
        eval_data.predict_data.predicted_labels,
        eval_data.predict_data.scores,
        eval_data.true_labels,
    )

    assert sorted(actual_eval_data_tpls, key=lambda t: t[0]) == sorted(
        expected_eval_data_tpls, key=lambda t: t[0]
    )


def test_load_eval_input_from_tsv(tsv_evaluate_data_path, eval_data):
    assert e._load_eval_input(tsv_evaluate_data_path) == eval_data
