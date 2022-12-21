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

from rdflib import URIRef, Graph

from qualle.features.label_calibration.thesauri_label_calibration import \
    Thesaurus
from tests.features.label_calibration.test_thesauri_label_calibration import \
    common as c


def test_get_concepts_for_thesaurus(thesaurus):
    concepts = thesaurus.get_concepts_for_subthesaurus(c.SUBTHESAURUS_A)
    assert concepts == {c.CONCEPT_x0, c.CONCEPT_x1, c.CONCEPT_x2}

    concepts = thesaurus.get_concepts_for_subthesaurus(c.SUBTHESAURUS_B)
    assert concepts == {c.CONCEPT_x1, c.CONCEPT_x2}

    concepts = thesaurus.get_concepts_for_subthesaurus(c.SUBTHESAURUS_C)
    assert concepts == {c.CONCEPT_x2}


def test_get_concepts_for_thesaurus_not_found_returns_empty(thesaurus):
    assert thesaurus.get_concepts_for_subthesaurus(
        URIRef(c.CONCEPT_URI_PREFIX + '/' + c.CONCEPT_INVALID)) == set()


def test_get_all_subthesauri(thesaurus):
    assert set(thesaurus.get_all_subthesauri()) == {
        c.SUBTHESAURUS_A, c.SUBTHESAURUS_B, c.SUBTHESAURUS_C
    }


def test_get_all_subthesauri_with_empty_thesaurus_returns_empty():
    t = Thesaurus(
        graph=Graph(),
        subthesaurus_type_uri=c.DUMMY_SUBTHESAURUS_TYPE,
        concept_type_uri=c.DUMMY_CONCEPT_TYPE,
        concept_uri_prefix=c.CONCEPT_URI_PREFIX
    )
    assert not t.get_all_subthesauri()


def test_extract_concept_from_uri_ref(thesaurus):
    assert thesaurus.extract_concept_id_from_uri_ref(
        URIRef(c.CONCEPT_URI_PREFIX + '/' + c.CONCEPT_x0)) == c.CONCEPT_x0
