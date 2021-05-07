FROM python:3.8-slim-buster
LABEL maintainer="AutoSE <AutoSE@zbw.eu>"

ARG PIP_INDEX_URL
ARG PIPENV_PYPI_MIRROR

RUN pip install --upgrade pip --no-cache-dir
RUN pip install pipenv gunicorn==20.1.* uvicorn[standard]==0.13.* --no-cache-dir

COPY Pipfile Pipfile.lock /app/

WORKDIR /app

RUN pipenv install --system \
	&& pip uninstall -y pipenv \
	&& rm -rf /root/.cache/pip*


COPY qualle qualle

CMD gunicorn "qualle.interface.rest:create_app()"  -b 0.0.0.0 -k "uvicorn.workers.UvicornWorker"
