#  Copyright (c) 2021 ZBW  – Leibniz Information Centre for Economics
#
#  This file is part of qualle.
#
#  qualle is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  qualle is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with qualle.  If not, see <http://www.gnu.org/licenses/>.
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
