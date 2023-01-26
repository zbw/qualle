#  Copyright 2021-2023 ZBW – Leibniz Information Centre for Economics
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
from rdflib import Graph, RDF, URIRef
from rdflib.namespace import SKOS

from qualle.features.label_calibration.thesauri import (
    LabelCountForSubthesauriTransformer,
    Thesaurus,
)
from tests.features.label_calibration.test_thesauri import common as c


@pytest.fixture
def graph():
    c_x0 = URIRef(f"{c.CONCEPT_URI_PREFIX}/{c.CONCEPT_x0}")
    c_x1 = URIRef(f"{c.CONCEPT_URI_PREFIX}/{c.CONCEPT_x1}")
    c_x2 = URIRef(f"{c.CONCEPT_URI_PREFIX}/{c.CONCEPT_x2}")

    g = Graph()
    for s in (c.SUBTHESAURUS_A, c.SUBTHESAURUS_B, c.SUBTHESAURUS_C):
        g.add((s, RDF.type, c.DUMMY_SUBTHESAURUS_TYPE))
    for concept in (c_x0, c_x1, c_x2):
        g.add((concept, RDF.type, c.DUMMY_CONCEPT_TYPE))

    g.add(
        (
            c.SUBTHESAURUS_A,
            SKOS.narrower,
            c.SUBTHESAURUS_C,
        )
    )

    g.add(
        (
            c.SUBTHESAURUS_A,
            SKOS.narrower,
            c_x0,
        )
    )
    g.add(
        (
            c.SUBTHESAURUS_A,
            SKOS.narrower,
            c_x1,
        )
    )
    g.add(
        (
            c.SUBTHESAURUS_B,
            SKOS.narrower,
            c_x1,
        )
    )
    g.add(
        (
            c.SUBTHESAURUS_B,
            SKOS.narrower,
            c_x2,
        )
    )
    g.add(
        (
            c.SUBTHESAURUS_C,
            SKOS.narrower,
            c_x2,
        )
    )
    return g


@pytest.fixture
def thesaurus(graph):
    return Thesaurus(
        graph=graph,
        subthesaurus_type_uri=c.DUMMY_SUBTHESAURUS_TYPE,
        concept_type_uri=c.DUMMY_CONCEPT_TYPE,
        concept_uri_prefix=c.CONCEPT_URI_PREFIX,
    )


@pytest.fixture
def transformer(thesaurus):
    transformer = LabelCountForSubthesauriTransformer(use_sparse_count_matrix=False)
    transformer.init(
        thesaurus=thesaurus, subthesauri=[c.SUBTHESAURUS_A, c.SUBTHESAURUS_B]
    )
    return transformer


@pytest.fixture
def uninitialized_transformer():
    return LabelCountForSubthesauriTransformer(use_sparse_count_matrix=False)
