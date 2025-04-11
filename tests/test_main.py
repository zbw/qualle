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
import pytest
import qualle.main as main
import sys
import subprocess


@pytest.fixture
def mdl_path(tmp_path):
    fp = tmp_path / "model"
    fp.write_text("")
    return fp


def test_main_with_mock(mocker, monkeypatch):
    mock_cli_entrypoint = mocker.patch("qualle.main.cli_entrypoint")
    monkeypatch.setattr(
        sys, "argv", ["main.py", "rest", str(mdl_path), "--host=x", "--port=9000"]
    )
    main.main()
    mock_cli_entrypoint.assert_called_once()


def test_main_with_error():
    result = subprocess.run(
        [sys.executable, "main.py", "rest", str(mdl_path), "--host=x", "--port=9000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert "can't open file" in result.stderr
