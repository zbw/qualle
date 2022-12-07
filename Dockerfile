FROM python:3.8-slim-buster
LABEL maintainer="AutoSE <AutoSE@zbw.eu>"

ARG POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip --no-cache-dir
RUN pip install poetry gunicorn==20.1.* "uvicorn[standard]>=0.14.*,<0.15" --no-cache-dir

COPY pyproject.toml poetry.lock /app/

WORKDIR /app

RUN poetry install --no-root --without dev \
	&& pip uninstall -y poetry \
	&& rm -rf /root/.cache/pypoetry


COPY qualle qualle

CMD gunicorn "qualle.interface.rest:create_app()"  -b 0.0.0.0 -k "uvicorn.workers.UvicornWorker"
