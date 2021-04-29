# Qualle

Implementation of Qualle Framework as proposed in the paper
[``Content-Based Quality Estimation for Automatic Subject Indexing of Short Texts
under Precision and Recall Constraints``](https://doi.org/10.1007/978-3-030-00066-0_1) from M. Toepfer.

The framework allows to train a model which can be used to predict
the quality of the result of the application of a multi-label classification (MLC) 
method on a  document. Currently, only the
[Recall](https://en.wikipedia.org/wiki/Precision_and_recall) 
is predicted for a document, but in principle
any document-level quality estimation (like prediction of precision) can be 
implemented in the future.

Qualle provides a Command-line interface to train
and evaluate models. Beyond that, a REST Webservice for predicting
the Recall of a MLC result is provided.

### Command line interface (CLI)
In order to run the CLI, you must install the packages from the Pipfile.
The interface is than accessible from the module ``qualle.main``. To
see the help message, run (inside the Qualle directory)

``python -m qualle.main -h``


### Train
You must provide a train data file in order to train a model. 
This file is a tabular-separated file (tsv) in the format (tabular is marked with ``\t``)

```document-content\tpredicted_labels_with_scores\ttrue_labels```

where
- ``document-content`` is a string describing the content of the document
(more precise: the string on which the MLC method is trained on), e.g. the title
- ``predicted_labels_with_scores`` is a comma-separated list of pairs ``predicted_label:confidence-score``
This is in principle the output of the MLC method.
- ``true_labels`` is a comma-separated list of true labels (ground truth)

For example, a row in the data file could look like 

``Optimal investment policy of the regulated firm\tConcept0:0.5,Concept1:1\tConcept0,Concept3``

To train a model, use the ``main`` module inside ``qualle``, e.g.:

``python -m qualle.main train /path/to/train_data_file /path/to/output/model``

It is also possible to use Label Calibration using the Subthesauri of a Thesaurus (like STW)
as categories (please read the paper for more explanation). Consult the help for the required options.

### Evaluate
You must provide a test data file and the path to a trained model in order to evaluate a model.
The test data file has the same format as the train data file. Metrics
like the [Explained Variation](https://en.wikipedia.org/wiki/Explained_variation) are printed out, describing the quality
of the Recall Prediction (please consult the paper for more information).

### REST interface
To perform the prediction on a MLC result, a REST interface can be started. 
[uvicorn](https://www.uvicorn.org/) is used as HTTP Server. You can also use any
ASGI Server implementation and create the ASGI app directly with the method
``qualle.interface.rest.create_app``. You need to provide the environment variable
MODEL_FILE with the path to the model (see ``qualle.interface.config.RESTSettings``).

The REST Endpoint expects a HTTP POST with the result of a MLC for a list of documents
as body. The format is JSON as specified in ``qualle/openapi.json``. You can also use
the Swagger UI accessible at ``http://address_of_server/docs`` to play a bit around.
