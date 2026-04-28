#  Copyright 2021-2026 ZBW – Leibniz Information Centre for Economics
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

FROM python:3.10-slim-trixie@sha256:e0ba03b9ca4e5736a351687c33f719d52991117d4586ce972548760f973e5b2e
LABEL maintainer="AutoSE <AutoSE@zbw.eu>"

ARG POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip --no-cache-dir
RUN pip install poetry gunicorn==25.3.* "uvicorn[standard]==0.40" --no-cache-dir

COPY pyproject.toml poetry.lock README.md  /app/

WORKDIR /app

COPY qualle qualle

RUN poetry install --without dev,ci \
	&& pip uninstall -y poetry \
	&& rm -rf /root/.cache/pypoetry

CMD ["gunicorn", "qualle.interface.rest:create_app()", "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker"]
