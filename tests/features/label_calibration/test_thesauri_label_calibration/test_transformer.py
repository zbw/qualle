#  Copyright 2021-2022 ZBW – Leibniz Information Centre for Economics
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

import numpy as np
import pytest
from rdflib import URIRef
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError

from qualle.features.label_calibration.thesauri_label_calibration import \
    LabelCountForSubthesauriTransformer
from tests.features.label_calibration.test_thesauri_label_calibration import \
    common as c


def test_fit_stores_concept_to_subthesauri_mapping(transformer):
    transformer.fit()

    assert hasattr(transformer, 'mapping_')
    assert transformer.mapping_ == {
        c.CONCEPT_x0: [True, False],
        c.CONCEPT_x1: [True] * 2,
        c.CONCEPT_x2: [True] * 2
    }


def test_fit_uses_all_subthesauri_if_no_subthesauri_passed(graph):
    transformer = LabelCountForSubthesauriTransformer(
        graph=graph,
        subthesaurus_type_uri=c.DUMMY_SUBTHESAURUS_TYPE,
        concept_type_uri=c.DUMMY_CONCEPT_TYPE,
        concept_uri_prefix=c.CONCEPT_URI_PREFIX,
        use_sparse_count_matrix=False
    )
    transformer.fit()
    assert hasattr(transformer, 'mapping_')
    x0_row = transformer.mapping_.get(c.CONCEPT_x0)
    x1_row = transformer.mapping_.get(c.CONCEPT_x1)
    x2_row = transformer.mapping_.get(c.CONCEPT_x2)

    assert all(
        (type(x0_row) == list, type(x1_row) == list, type(x2_row) == list)
    )
    assert all(
        (len(x0_row) == 3, len(x1_row) == 3, len(x2_row) == 3)
    )
    # We dont know the order of the subthesaui extracted. Therefore we check
    # if the expected columns are present.
    col0 = [x0_row[0], x1_row[0], x2_row[0]]
    col1 = [x0_row[1], x1_row[1], x2_row[1]]
    col2 = [x0_row[2], x1_row[2], x2_row[2]]

    cols = [col0, col1, col2]

    assert [True, True, True] in cols
    assert [False, True, True] in cols
    assert [False, False, True] in cols


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


def test_transform_returns_sparse_count_matrix(transformer):
    transformer.use_sparse_count_matrix = True
    transformer.fit()
    count_matrix = transformer.transform([
                [c.CONCEPT_x0, c.CONCEPT_INVALID], [c.CONCEPT_x0, c.CONCEPT_x2]
            ])
    assert issparse(count_matrix)
    assert (count_matrix.toarray() == np.array([[1, 0], [2, 1]])).all()


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
