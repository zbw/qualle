FROM python:3.8-slim-buster
LABEL maintainer="AutoSE <AutoSE@zbw.eu>"

ARG POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip --no-cache-dir
RUN pip install poetry gunicorn==22.0.* "uvicorn[standard]==0.22" --no-cache-dir

COPY pyproject.toml poetry.lock README.md  /app/

WORKDIR /app

COPY qualle qualle

RUN poetry install --without dev,ci \
	&& pip uninstall -y poetry \
	&& rm -rf /root/.cache/pypoetry

CMD gunicorn "qualle.interface.rest:create_app()"  -b 0.0.0.0 -k "uvicorn.workers.UvicornWorker"
