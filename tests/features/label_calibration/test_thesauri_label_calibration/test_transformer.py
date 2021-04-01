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
from rdflib import URIRef
from sklearn.exceptions import NotFittedError

from tests.features.label_calibration.test_thesauri_label_calibration import \
    common as c


def test_fit_stores_concept_to_subthesauri_mapping(transformer):
    transformer.fit()

    assert hasattr(transformer, 'mapping_')
    assert transformer.mapping_ == defaultdict(list, {
        c.CONCEPT_x0: [True, False],
        c.CONCEPT_x1: [True] * 2,
        c.CONCEPT_x2: [True] * 2
    })


def test_transform_without_fit_raises_exc(transformer):
    with pytest.raises(NotFittedError):
        transformer.transform(None)


def test_transform_returns_count_matrix(transformer):
    transformer.fit()
    assert (
            transformer.transform([
                [c.CONCEPT_x0, c.CONCEPT_INVALID], [c.CONCEPT_x0, c.CONCEPT_x2]
            ])
            == np.array([[1, 0], [2, 1]])
    ).all()


def test_get_concepts_for_thesaurus(transformer):
    transformer.fit()
    concepts = transformer._get_concepts_for_thesaurus(c.SUBTHESAURUS_A)
    assert concepts == {c.CONCEPT_x0, c.CONCEPT_x1, c.CONCEPT_x2}

    concepts = transformer._get_concepts_for_thesaurus(c.SUBTHESAURUS_B)
    assert concepts == {c.CONCEPT_x1, c.CONCEPT_x2}

    concepts = transformer._get_concepts_for_thesaurus(c.SUBTHESAURUS_C)
    assert concepts == {c.CONCEPT_x2}


def test_get_concepts_for_thesaurus_not_found_returns_empty(transformer):
    transformer.fit()
    assert transformer._get_concepts_for_thesaurus(
        URIRef(c.CONCEPT_URI_PREFIX + '/' + c.CONCEPT_INVALID)) == set()


def test_extract_concept_from_uri_ref(transformer):
    transformer.fit()
    assert transformer._extract_concept_id_from_uri_ref(
        URIRef(c.CONCEPT_URI_PREFIX + '/' + c.CONCEPT_x0)) == c.CONCEPT_x0
