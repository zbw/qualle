#  Copyright (c) 2021 ZBW  – Leibniz Information Centre for Economics
#
#  This file is part of qualle.
#
#  qualle is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  qualle is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with qualle.  If not, see <http://www.gnu.org/licenses/>.
from collections import defaultdict

import numpy as np
import pytest
from rdflib import URIRef, Graph, RDF
from rdflib.namespace import SKOS
from sklearn.exceptions import NotFittedError

from qualle.thesaurus import LabelCountForSubthesauriTransformer

DUMMY_SUBTHESAURUS_TYPE = URIRef('http://type/Thsys')
DUMMY_CONCEPT_TYPE = URIRef('http://type/Descriptor')

SUBTHESAURUS_A = URIRef('http://thsys/A')
SUBTHESAURUS_B = URIRef('http://thsys/B')
SUBTHESAURUS_C = URIRef('http://thsys/C')


CONCEPT_x0 = URIRef('http://concept/x0')
CONCEPT_x1 = URIRef('http://concept/x1')
CONCEPT_x2 = URIRef('http://concept/x2')


@pytest.fixture
def graph():
    g = Graph()
    for s in (SUBTHESAURUS_A, SUBTHESAURUS_B, SUBTHESAURUS_C):
        g.add((
            s,
            RDF.type,
            DUMMY_SUBTHESAURUS_TYPE))
    for c in (CONCEPT_x0, CONCEPT_x1, CONCEPT_x2):
        g.add((
            c,
            RDF.type,
            DUMMY_CONCEPT_TYPE))

    g.add((
        SUBTHESAURUS_A,
        SKOS.narrower,
        SUBTHESAURUS_C,
    ))

    g.add((
        SUBTHESAURUS_A,
        SKOS.narrower,
        CONCEPT_x0,
    ))
    g.add((
        SUBTHESAURUS_A,
        SKOS.narrower,
        CONCEPT_x1,
    ))
    g.add((
        SUBTHESAURUS_B,
        SKOS.narrower,
        CONCEPT_x1,
    ))
    g.add((
        SUBTHESAURUS_B,
        SKOS.narrower,
        CONCEPT_x2,
    ))
    g.add((
        SUBTHESAURUS_C,
        SKOS.narrower,
        CONCEPT_x2,
    ))
    return g


@pytest.fixture
def transformer(graph):
    return LabelCountForSubthesauriTransformer(
        graph=graph,
        subthesaurus_type_uri=DUMMY_SUBTHESAURUS_TYPE,
        concept_type_uri=DUMMY_CONCEPT_TYPE,
        subthesauri=[SUBTHESAURUS_A, SUBTHESAURUS_B]
    )


def test_fit_stores_concept_to_subthesauri_mapping(transformer):
    transformer.fit()

    assert hasattr(transformer, 'mapping_')
    assert transformer.mapping_ == defaultdict(list, {
        CONCEPT_x0: [True, False],
        CONCEPT_x1: [True] * 2,
        CONCEPT_x2: [True] * 2
    })


def test_transform_without_fit_raises_exc(transformer):
    with pytest.raises(NotFittedError):
        transformer.transform(None)


def test_transform_returns_count_matrix(transformer):
    transformer.fit()
    assert (
            transformer.transform([[CONCEPT_x0], [CONCEPT_x0, CONCEPT_x2]])
            == np.array([[1, 0], [2, 1]])
    ).all()


def test_get_concepts_for_thesaurus(transformer):
    concepts = transformer._get_concepts_for_thesaurus(SUBTHESAURUS_A)
    assert concepts == {CONCEPT_x0, CONCEPT_x1, CONCEPT_x2}

    concepts = transformer._get_concepts_for_thesaurus(SUBTHESAURUS_B)
    assert concepts == {CONCEPT_x1, CONCEPT_x2}

    concepts = transformer._get_concepts_for_thesaurus(SUBTHESAURUS_C)
    assert concepts == {CONCEPT_x2}


def test_get_concepts_for_thesaurus_not_found_returns_empty(transformer):
    assert transformer._get_concepts_for_thesaurus(
        URIRef('http://thsys/invalid')) == set()
