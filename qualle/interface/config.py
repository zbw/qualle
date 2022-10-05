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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseSettings
from pydantic.networks import AnyUrl
from qualle.features.confidence import ConfidenceFeatures
from qualle.features.text import TextFeatures


class RegressorSettings(BaseSettings):
    regressor_class: str
    params: Dict


class SubthesauriLabelCalibrationSettings(BaseSettings):
    thesaurus_file: Path
    subthesaurus_type: AnyUrl
    concept_type: AnyUrl
    concept_type_prefix: AnyUrl
    subthesauri: List[AnyUrl]
    use_sparse_count_matrix: bool = False


class FeaturesEnum(Enum):
    CONFIDENCE = ConfidenceFeatures
    TEXT = TextFeatures


class TrainSettings(BaseSettings):

    label_calibrator_regressor: RegressorSettings
    quality_estimator_regressor: RegressorSettings

    train_data_path: Path
    output_path: Path

    features: List[FeaturesEnum]

    subthesauri_label_calibration: Optional[
        SubthesauriLabelCalibrationSettings] = None

    should_debug: bool = False


class EvalSettings(BaseSettings):
    test_data_path: Path
    model_file: Path


class RESTSettings(BaseSettings):
    model_file: Path
    port: int = 8000
    host: str = '127.0.0.1'
