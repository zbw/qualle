# Qualle
[![Extended Tests](https://github.com/zbw/qualle/actions/workflows/extended.yml/badge.svg)](https://github.com/zbw/qualle/actions/workflows/extended.yml)
[![codecov](https://codecov.io/gh/zbw/qualle/branch/master/graph/badge.svg?token=ZE7OWKA83Q)](https://codecov.io/gh/zbw/qualle)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

This is an implementation of the Qualle framework as proposed in the paper
[1] and accompanying source code.

The framework allows to train a model which can be used to predict
the quality of the result of applying a multi-label classification (MLC)
method on a document. In this implementation, only the
[recall](https://en.wikipedia.org/wiki/Precision_and_recall)
is predicted for a document, but in principle
any document-level quality estimation (such as the prediction of precision)
can be implemented analogously.

Qualle provides a command-line interface to train
and evaluate models. In addition, a REST webservice for predicting
the recall of a MLC result is provided.


## Requirements

Python ``>= 3.9`` is required.

## Installation

Choose one of these installation methods:

### With pip
Qualle is available on [PyPI](https://pypi.org/) . You can install Qualle using pip:

``pip install qualle``

This will install a command line tool called `qualle` . You can call `qualle -h` to see the help message which will
display the available modes and options.

Note that it is generally recommended to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to avoid
 conflicting behaviour with the system package manager.

### From source
You also have the option to checkout the repository and install the packages from source. You need
[poetry](https://python-poetry.org) to perform the task:

```shell
# call inside the project directory
poetry install --without ci
```

### Docker
You can also use a Docker Image from the [Container Registry of Github](https://github.com/zbw/qualle/pkgs/container/qualle):

``docker pull ghcr.io/zbw/qualle``

Alternatively, you can use the Dockerfile included in this project to build a Docker image yourself. E.g.:

 ``docker build -t qualle .``

By default, a container built from this image launches a REST interface listening on ``0.0.0.0:8000``

You need to pass the model file (see below the section REST interface) per bind mount or volume to the docker container.
Beyond that, you need to specify the location of the model file with an
environment variable named `MDL_FILE`:

``docker run --rm -it --env MDL_FILE=/model -v /path/to/model:/model -p 8000:8000 ghcr.io/zbw/qualle``

[Gunicorn](https://gunicorn.org/) is used as HTTP Server. You can use the environment variable ``GUNICORN_CMD_ARGS`` to customize
Gunicorn settings, such as the number of worker processes to use:

``docker run --rm -it --env MDL_FILE=/model --env GUNICORN_CMD_ARGS="--workers 4" -v /path/to/model:/model -p 8000:8000 ghcr.io/zbw/qualle``

You can also use the Docker image to train or evaluate by using the Qualle command line tool:

```shell
docker run --rm -it -v \
 /path/to/train_data_file:/train_data_file -v /path/to/model_dir:/mdl_dir ghcr.io/zbw/qualle \
 qualle train /train_data_file /mdl_dir/model
 ```

The Qualle command line tool is not available for the release 0.1.0 and 0.1.1. For these releases,
you need to call the python module ``qualle.main`` instead:

```shell
docker run --rm -it -v \
 /path/to/train_data_file:/train_data_file -v /path/to/model_dir:/model_dir ghcr.io/zbw/qualle:0.1.1 \
 python -m qualle.main train /train_data_file /model_dir/model
```

## Usage

### Input data
In order to train a model, evaluate a model or predict the quality of an MLC result
you have to provide data.

This can be a tabular-separated file (tsv) in the format (tabular is marked with ``\t``)

```document-content\tpredicted_labels_with_scores\ttrue_labels```

where
- ``document-content`` is a string describing the content of the document
(more precisely: the string on which the MLC method is trained), e.g. the title
- ``predicted_labels_with_scores`` is a comma-separated list of pairs ``predicted_label:confidence-score``
(this is basically the output of the MLC method)
- ``true_labels`` is a comma-separated list of true labels (ground truth)

Note that you can omit the ``true_labels`` section if you only want to predict the
quality of the MLC result.

For example, a row in the data file could look like this:

``Optimal investment policy of the regulated firm\tConcept0:0.5,Concept1:1\tConcept0,Concept3``

For those who use an MLC method via the toolkit [Annif](https://github.com/NatLibFi/annif) for automated subject indexing:
You can alternatively specify a
[full-text document corpus](https://github.com/NatLibFi/Annif/wiki/Document-corpus-formats/70a8f079313e872ed513a4bff1747c604b5781a7)
 combined with the result of the Annif index method (tested with Annif version 0.59) applied on the corpus.
This is a  folder with three files per document:

* ``doc.annif`` : result of Annif index method
* ``doc.tsv`` : ground truth
* ``doc.txt`` : document content

As above, you may omit the ``doc.tsv`` if you just want to
predict the quality of the MLC result.

### Train
To train a model, use the ``train`` mode, e.g.:

``qualle train /path/to/train_data_file /path/to/output/model``

It is also possible to use label calibration (comparison of predicted vs actual labels) using the subthesauri of a thesaurus (such as the [STW](http://zbw.eu/stw/version/latest/about))
as categories (please read the paper for more explanations). Consult the help (see above) for the required options.

### Evaluate
You must provide test data and the path to a trained model in order to evaluate that model. Metrics
such as the [explained variation](https://en.wikipedia.org/wiki/Explained_variation) are printed out, describing the quality
of the recall prediction (please consult the paper for more information).

### REST interface
To perform the prediction on a MLC result, a REST interface can be started.
[uvicorn](https://www.uvicorn.org/) is used as HTTP server. You can also use any
ASGI server implementation and create the ASGI app directly with the method
``qualle.interface.rest.create_app``. You need to provide the environment variable
MDL_FILE with the path to the model (see ``qualle.interface.config.RESTSettings``).

The REST endpoint expects a HTTP POST with the result of a MLC for a list of documents
as body. The format is JSON as specified in ``qualle/openapi.json``. You can also use
the Swagger UI accessible at ``http://address_of_server/docs`` to play around a bit.


## Contribute

Contributions via pull requests are welcome. Please create an issue beforehand
to explain and discuss the reasons for the respective contribution.

qualle code should follow the [Black style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html).
The Black tool is included as a development dependency; you can run `black .` in the project root to autoformat code. There is also the possibility of doing this with a Git Pre-Commit hook script. It is already configured in the `.pre-commit-config.yaml` file. The [pre-commit](https://pre-commit.com/#installation) tool has been included as a development dependency. You would have to run the command `pre-commit install` inside your local virtual environment. Subsequently, the Black tool will automatically check the formatting of modified or new scripts after each time a `git commit` command is executed.

## References
[1] [Toepfer, Martin, and Christin Seifert. "Content-based quality estimation for automatic subject indexing of short texts under precision and recall constraints." International Conference on Theory and Practice of Digital Libraries. Springer, Cham, 2018., DOI 10.1007/978-3-030-00066-0_1](https://arxiv.org/abs/1806.02743)

## Context information
This code was created as part of the subject indexing automatization effort at [ZBW - Leibniz Information Centre for Economics](https://www.zbw.eu/en/). See [our homepage](https://www.zbw.eu/en/about-us/knowledge-organisation/automation-of-subject-indexing-using-methods-from-artificial-intelligence) for more information, publications, and contact details.
