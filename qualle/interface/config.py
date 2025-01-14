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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from typing_extensions import Annotated

from pydantic import (
    model_validator,
    FilePath,
    DirectoryPath,
    TypeAdapter,
    PlainValidator,
    AfterValidator,
)
from pydantic_settings import BaseSettings
from pydantic.networks import AnyUrl
from qualle.features.confidence import ConfidenceFeatures
from qualle.features.text import TextFeatures

# From pydantic v2 onwards, AnyUrl object does not inherit from a string class.
# The following code block performs validation on a pydantic AnyUrl object
# as if it were a string.  Another problem is that a trailing '/' character is also
# appended in pydantic v2 and it is being removed in the code block given below.

AnyUrlAdapter = TypeAdapter(AnyUrl)
HttpUrlStr = Annotated[
    str,
    PlainValidator(lambda x: AnyUrlAdapter.validate_strings(x)),
    AfterValidator(lambda x: str(x).rstrip("/")),
]


FileOrDirPath = Union[FilePath, DirectoryPath]


class RegressorSettings(BaseSettings):
    regressor_class: str
    params: Dict


class SubthesauriLabelCalibrationSettings(BaseSettings):
    thesaurus_file: FilePath
    subthesaurus_type: HttpUrlStr
    concept_type: HttpUrlStr
    concept_type_prefix: HttpUrlStr
    subthesauri: List[HttpUrlStr]
    use_sparse_count_matrix: bool = False


class FeaturesEnum(Enum):
    CONFIDENCE = ConfidenceFeatures
    TEXT = TextFeatures


class TrainSettings(BaseSettings):

    label_calibrator_regressor: RegressorSettings
    quality_estimator_regressor: RegressorSettings

    train_data_path: FileOrDirPath
    output_path: Path

    features: List[FeaturesEnum]

    subthesauri_label_calibration: Optional[SubthesauriLabelCalibrationSettings] = None

    should_debug: bool = False


class EvalSettings(BaseSettings):
    test_data_path: FileOrDirPath
    mdl_file: FilePath


class PredictSettings(BaseSettings):
    predict_data_path: FileOrDirPath
    mdl_file: FilePath
    output_path: Optional[Path] = None

    @model_validator(mode="after")
    def check_output_path_specified_for_input_file(self):
        predict_data_path = self.predict_data_path
        output_path = self.output_path
        if predict_data_path.is_file() and not output_path:
            raise ValueError(
                "output_path has to be specified if predict_data_path "
                "refers to a file"
            )
        return self


class RESTSettings(BaseSettings):
    mdl_file: FilePath
    port: int = 8000
    host: str = "127.0.0.1"
