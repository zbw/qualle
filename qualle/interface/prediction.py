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
from pathlib import Path
from typing import Optional, List, Any, Union

from qualle.interface.config import PredictSettings
from qualle.interface.data import tsv, annif
from qualle.interface.common import load_model
from qualle.models import PredictData
from qualle.utils import get_logger


def predict(settings: PredictSettings):
    logger = get_logger()
    path_to_predict_data = settings.predict_data_path
    path_to_model_file = settings.model_file
    output_path = settings.output_path
    model = load_model(str(path_to_model_file))
    io_handler = _get_predict_io_handler(
        predict_data_path=path_to_predict_data, output_path=output_path
    )
    predict_data = io_handler.load_predict_data()
    logger.info("Run predict with model:\n%s", model)

    scores = model.predict(predict_data)
    io_handler.store(scores)


class PredictTSVIOHandler:
    def __init__(self, predict_data_path: Path, output_path: Optional[Path] = None):
        self._predict_data_path = predict_data_path
        self._output_path = output_path

    def load_predict_data(self):
        return tsv.load_predict_input(self._predict_data_path)

    def store(self, data: List[Any]):
        with self._output_path.open("w") as f:
            for d in data:
                f.write(str(d) + "\n")


class PredictAnnifIOHandler:
    def __init__(self, predict_data_dir: Path):
        self._dir = predict_data_dir
        self._doc_ids = []

    def load_predict_data(self) -> PredictData:
        data = annif.load_predict_input(self._dir)
        self._doc_ids = data.document_ids
        return data.predict_data

    def store(self, data: List[Any]):
        quality_ests = zip(data, self._doc_ids)
        annif.store_quality_estimations(dir=self._dir, quality_ests=quality_ests)


def _get_predict_io_handler(
    predict_data_path: Path, output_path: Optional[Path]
) -> Union[PredictAnnifIOHandler, PredictTSVIOHandler]:
    if predict_data_path.is_dir():
        return PredictAnnifIOHandler(predict_data_path)
    return PredictTSVIOHandler(
        predict_data_path=predict_data_path, output_path=output_path
    )
