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

FROM python:3.10-slim-trixie@sha256:8769cc10045c67eac267537ff0950ce644eeebd9f0e88bec0604548a88f1f33d
LABEL maintainer="AutoSE <AutoSE@zbw.eu>"

ARG POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip --no-cache-dir
RUN pip install poetry gunicorn==23.0.* "uvicorn[standard]==0.38" --no-cache-dir

COPY pyproject.toml poetry.lock README.md  /app/

WORKDIR /app

COPY qualle qualle

RUN poetry install --without dev,ci \
	&& pip uninstall -y poetry \
	&& rm -rf /root/.cache/pypoetry

CMD ["gunicorn", "qualle.interface.rest:create_app()", "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker"]
