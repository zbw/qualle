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

from qualle.interface.config import (
    PredictSettings,
)
import qualle.interface.prediction as p
from qualle.models import PredictData
from tests.interface.common import URI_PREFIX


@pytest.fixture(autouse=True)
def mock_io(mocker):
    mocker.patch("qualle.interface.prediction.load_model")


@pytest.fixture
def predict_data():
    return PredictData(
        docs=[f"Title{i}" for i in range(20)],
        predicted_labels=[[f"concept{i}"] for i in range(20)],
        scores=[[i / 20] for i in range(20)],
    )


@pytest.fixture
def tsv_predict_data_path(tmp_path):
    tsv_file = tmp_path / "predict-docs.tsv"
    content = "\n".join(f"Title{i}\tconcept{i}:{i / 20}" for i in range(20))
    tsv_file.write_text(content)
    return tsv_file


@pytest.fixture
def annif_predict_data_dir(tmp_path):
    p_dir = tmp_path / "predict"
    p_dir.mkdir()
    for i in range(20):
        doc0 = p_dir / f"doc{i}.txt"
        doc0.write_text(f"Title{i}")
        scores0 = p_dir / f"doc{i}.annif"
        scores0.write_text(f"<{URI_PREFIX}concept{i}>\tlabel0\t{i / 20}\n")
    return p_dir


def test_predict_stores_scores_from_model(tsv_predict_data_path, tmp_path, model_path):
    output_path = tmp_path / "qualle.txt"
    settings = PredictSettings(
        predict_data_path=tsv_predict_data_path,
        model_file=model_path,
        output_path=output_path,
    )
    mock_model = p.load_model.return_value
    mock_model.predict.side_effect = lambda p_data: map(lambda s: s[0], p_data.scores)

    p.predict(settings)

    assert output_path.read_text().rstrip("\n") == "\n".join(
        [str(x / 20) for x in range(20)]
    )


def test_predict_with_annif_data_stores_scores_from_model(
    annif_predict_data_dir, tmp_path, model_path
):
    settings = PredictSettings(
        predict_data_path=annif_predict_data_dir,
        model_file=model_path,
    )
    mock_model = p.load_model.return_value
    mock_model.predict.side_effect = lambda p_data: map(lambda s: s[0], p_data.scores)

    p.predict(settings)

    for i in range(20):
        fp = annif_predict_data_dir / f"doc{i}.qualle"
        assert fp.exists()
        assert fp.read_text() == str(i / 20), f"fail for {fp}"
