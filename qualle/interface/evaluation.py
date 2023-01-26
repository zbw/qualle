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

from qualle.evaluate import Evaluator
from qualle.interface.config import EvalSettings
from qualle.interface.common import load_model
from qualle.models import EvalData
from qualle.utils import get_logger

from qualle.interface.data import annif
from qualle.interface.data import tsv


def evaluate(settings: EvalSettings):
    logger = get_logger()
    path_to_test_data = settings.test_data_path
    path_to_model_file = settings.model_file
    model = load_model(str(path_to_model_file))
    logger.info("Run evaluation with model:\n%s", model)
    test_input = _load_eval_input(path_to_test_data)
    ev = Evaluator(test_input, model)
    eval_data = ev.evaluate()
    logger.info("\nScores:")
    for metric, score in eval_data.items():
        logger.info(f"{metric}: {score}")


def _load_eval_input(eval_data_path: Path) -> EvalData:
    if eval_data_path.is_dir():
        eval_data = annif.load_predict_train_input(eval_data_path).predict_train_data
    else:
        eval_data = tsv.load_predict_train_input(eval_data_path)

    return eval_data
