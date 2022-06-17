# Qualle
![CI](https://github.com/zbw/qualle/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/zbw/qualle/branch/master/graph/badge.svg?token=ZE7OWKA83Q)](https://codecov.io/gh/zbw/qualle)

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

### Command line interface (CLI)
In order to run the CLI, you must install the packages from the Pipfile.
The interface is then accessible from the module ``qualle.main``. To
see the help message, run (inside the Qualle directory)

``python -m qualle.main -h``


### Train
In order to train a model you have to provide a training data file.
This file has to be a tabular-separated file (tsv) in the format (tabular is marked with ``\t``)

```document-content\tpredicted_labels_with_scores\ttrue_labels```

where
- ``document-content`` is a string describing the content of the document
(more precisely: the string on which the MLC method is trained), e.g. the title
- ``predicted_labels_with_scores`` is a comma-separated list of pairs ``predicted_label:confidence-score``
(this is basically the output of the MLC method)
- ``true_labels`` is a comma-separated list of true labels (ground truth)

For example, a row in the data file could look like this:

``Optimal investment policy of the regulated firm\tConcept0:0.5,Concept1:1\tConcept0,Concept3``

To train a model, use the ``main`` module inside ``qualle``, e.g.:

``python -m qualle.main train /path/to/train_data_file /path/to/output/model``

It is also possible to use label calibration using the subthesauri of a thesaurus (such as the [STW](http://zbw.eu/stw/version/latest/about))
as categories (please read the paper for more explanations). Consult the help (see above) for the required options.

### Evaluate
You must provide a test data file and the path to a trained model in order to evaluate that model.
The test data file has the same format as the training data file. Metrics
such as the [explained variation](https://en.wikipedia.org/wiki/Explained_variation) are printed out, describing the quality
of the recall prediction (please consult the paper for more information).

### REST interface
To perform the prediction on a MLC result, a REST interface can be started. 
[uvicorn](https://www.uvicorn.org/) is used as HTTP server. You can also use any
ASGI server implementation and create the ASGI app directly with the method
``qualle.interface.rest.create_app``. You need to provide the environment variable
MODEL_FILE with the path to the model (see ``qualle.interface.config.RESTSettings``).

The REST endpoint expects a HTTP POST with the result of a MLC for a list of documents
as body. The format is JSON as specified in ``qualle/openapi.json``. You can also use
the Swagger UI accessible at ``http://address_of_server/docs`` to play around a bit.

### Deployment with Docker
You can use the Dockerfile included in this project to build a Docker image. E.g.:

 ``docker build -t qualle .``

Per default, gunicorn is used to run the REST interface on ``0.0.0.0:8000``
You need to pass the required settings per environment variable. E.g.:

``docker run --rm -it --env model_file=/model -v /path/to/model:/model -p 8000:8000 qualle``

Of course you can also use the Docker image to train or evaluate by using a 
different command as input to [docker run](https://docs.docker.com/engine/reference/run/#general-form).

## References
[1] [Toepfer, Martin, and Christin Seifert. "Content-based quality estimation for automatic subject indexing of short texts under precision and recall constraints." International Conference on Theory and Practice of Digital Libraries. Springer, Cham, 2018., DOI 10.1007/978-3-030-00066-0_1](https://arxiv.org/abs/1806.02743)

## Context information
This code was created as part of the subject indexing automatization effort at [ZBW - Leibniz Information Centre for Economics](https://www.zbw.eu/en/). See [our homepage](https://www.zbw.eu/en/about-us/key-activities/automated-subject-indexing) for more information, publications, and contact details.
