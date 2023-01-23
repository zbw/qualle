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

import pytest
from pydantic import ValidationError

from qualle.interface.config import PredictSettings


def test_predict_settings_input_file_but_no_output_raises_exc(tmp_path):
    fp = tmp_path / "fp.tsv"
    fp.write_text("t\tc:0\tc")

    with pytest.raises(ValidationError):
        PredictSettings(predict_data_path=fp, model=tmp_path / "model")
