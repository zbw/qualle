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
import pytest
from rdflib import Graph, RDF, URIRef
from rdflib.namespace import SKOS

from qualle.features.label_calibration.thesauri_label_calibration import \
    LabelCountForSubthesauriTransformer
from tests.features.label_calibration.test_thesauri_label_calibration import \
    common as c


@pytest.fixture
def graph():
    c_x0 = URIRef(f'{c.CONCEPT_URI_PREFIX}/{c.CONCEPT_x0}')
    c_x1 = URIRef(f'{c.CONCEPT_URI_PREFIX}/{c.CONCEPT_x1}')
    c_x2 = URIRef(f'{c.CONCEPT_URI_PREFIX}/{c.CONCEPT_x2}')

    g = Graph()
    for s in (c.SUBTHESAURUS_A, c.SUBTHESAURUS_B, c.SUBTHESAURUS_C):
        g.add((
            s,
            RDF.type,
            c.DUMMY_SUBTHESAURUS_TYPE))
    for concept in (c_x0, c_x1, c_x2):
        g.add((
            concept,
            RDF.type,
            c.DUMMY_CONCEPT_TYPE))

    g.add((
        c.SUBTHESAURUS_A,
        SKOS.narrower,
        c.SUBTHESAURUS_C,
    ))

    g.add((
        c.SUBTHESAURUS_A,
        SKOS.narrower,
        c_x0,
    ))
    g.add((
        c.SUBTHESAURUS_A,
        SKOS.narrower,
        c_x1,
    ))
    g.add((
        c.SUBTHESAURUS_B,
        SKOS.narrower,
        c_x1,
    ))
    g.add((
        c.SUBTHESAURUS_B,
        SKOS.narrower,
        c_x2,
    ))
    g.add((
        c.SUBTHESAURUS_C,
        SKOS.narrower,
        c_x2,
    ))
    return g


@pytest.fixture
def transformer(graph):
    return LabelCountForSubthesauriTransformer(
        graph=graph,
        subthesaurus_type_uri=c.DUMMY_SUBTHESAURUS_TYPE,
        concept_type_uri=c.DUMMY_CONCEPT_TYPE,
        subthesauri=[c.SUBTHESAURUS_A, c.SUBTHESAURUS_B],
        concept_uri_prefix=c.CONCEPT_URI_PREFIX,
        use_sparse_count_matrix=False
    )
