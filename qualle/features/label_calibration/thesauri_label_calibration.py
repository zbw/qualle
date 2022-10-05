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

import logging
from functools import lru_cache
from typing import List, Set, Optional

import numpy as np
from rdflib import URIRef, Graph
from rdflib.namespace import SKOS, RDF
from scipy.sparse import coo_matrix
from sklearn.base import TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted
from stwfsapy import thesaurus

from qualle.features.base import Features
from qualle.features.label_calibration.base import AbstractLabelCalibrator
from qualle.label_calibration.category import MultiCategoryLabelCalibrator
from qualle.models import Labels, Documents, LabelCalibrationData, Matrix
from qualle.utils import get_logger


class LabelCountForSubthesauriTransformer(TransformerMixin):
    # Do not inherit from BaseEstimator to avoid refitting requirement when
    # transformer gets cloned (e.g. in cross-val-predict)
    """Compute count of labels per Subthesauri for a given RDF Graph."""
    def __init__(
            self,
            graph: Graph,
            subthesaurus_type_uri: URIRef,
            concept_type_uri: URIRef,
            concept_uri_prefix: str,
            use_sparse_count_matrix: bool,
            subthesauri: Optional[List[URIRef]] = None
    ):
        self.graph = graph
        self.subthesaurus_type_uri = subthesaurus_type_uri
        self.concept_type_uri = concept_type_uri
        self.subthesauri = subthesauri
        self.use_sparse_count_matrix = use_sparse_count_matrix
        self.concept_uri_prefix = concept_uri_prefix

    def fit(self, X=None, y=None):
        self.mapping_ = dict()
        self.logger_ = get_logger()
        self.concept_uri_prefix_len_ = 1 + len(
            self.concept_uri_prefix.rstrip('/')
        )
        if self.subthesauri:
            self.subthesauri_ = self.subthesauri
        else:
            self.subthesauri_ = list(thesaurus.extract_by_type_uri(
                self.graph,
                self.subthesaurus_type_uri,
            ))
        subthesauri_len = len(self.subthesauri_)

        for idx, s in enumerate(self.subthesauri_):
            concepts = self._get_concepts_for_thesaurus(s)
            for c in concepts:
                if c not in self.mapping_:
                    self.mapping_[c] = [False] * subthesauri_len
                self.mapping_[c][idx] = True
        return self

    def transform(self, X: List[Labels]) -> Matrix:
        """Transform rows of concepts to 2-dimensional count array

        Each row in the result matrix contains in each column the total
        amount of labels per subthesauri. The subthesauri are given
        in the same order as in the list passed to the constructor.
        """
        check_is_fitted(self)
        if self.use_sparse_count_matrix:
            values = []
            row_inds = []
            col_inds = []
        else:
            count_matrix = np.zeros((len(X), len(self.subthesauri_)))
        for row_idx, row in enumerate(X):
            subthesauri_counts = self._extract_subthesauri_counts(row)
            if self.use_sparse_count_matrix:
                for col_idx, count in enumerate(subthesauri_counts):
                    if count:
                        values.append(count)
                        row_inds.append(row_idx)
                        col_inds.append(col_idx)
            else:
                count_matrix[row_idx] = subthesauri_counts

        if self.use_sparse_count_matrix:
            return coo_matrix(
                (values, (row_inds, col_inds)),
                shape=(len(X), len(self.subthesauri_))
            )
        else:
            return count_matrix

    def _extract_subthesauri_counts(self, labels: Labels):
        subthesauri_counts = [0] * len(self.subthesauri_)
        for concept in labels:
            subthesauri_membership = self.mapping_.get(concept)
            if not subthesauri_membership:
                self.logger_.warning(
                    'Concept "%s" not found in concept map. '
                    'Seems to be invalid for this thesaurus.',
                    concept
                )
            else:
                for j, is_in_subthesauri in enumerate(subthesauri_membership):
                    if is_in_subthesauri:
                        subthesauri_counts[j] = subthesauri_counts[j] + 1
        return subthesauri_counts

    @lru_cache(maxsize=1000)
    def _get_concepts_for_thesaurus(self, thesaurus: URIRef) -> Set:
        concepts = set()
        for x in self.graph[thesaurus:SKOS.narrower]:
            if (x, RDF.type, self.subthesaurus_type_uri) in self.graph:
                concepts_from_subthesaurus = self._get_concepts_for_thesaurus(
                    x)
                concepts = concepts.union(concepts_from_subthesaurus)
            elif (x, RDF.type, self.concept_type_uri) in self.graph:
                concepts.add(self._extract_concept_id_from_uri_ref(x))
            else:
                logging.warning('unknown narrower type %s', str(x))
        return concepts

    def _extract_concept_id_from_uri_ref(self,  concept_uri: URIRef):
        return concept_uri.toPython()[self.concept_uri_prefix_len_:]


class ThesauriLabelCalibrator(AbstractLabelCalibrator):

    def __init__(
            self, transformer: LabelCountForSubthesauriTransformer,
            regressor_class=ExtraTreesRegressor,
            regressor_params=None
    ):
        self.transformer = transformer
        self.regressor_class = regressor_class
        self.regressor_params = regressor_params

    def fit(self, X: Documents, y: List[Labels]):
        self.calibrator_ = MultiCategoryLabelCalibrator(
            regressor_class=self.regressor_class,
            regressor_params=self.regressor_params
        )
        y_transformed = self.transformer.transform(y)
        self.calibrator_.fit(X, y_transformed)
        return self

    def predict(self, X: Documents):
        check_is_fitted(self)
        return self.calibrator_.predict(X)


class ThesauriLabelCalibrationFeatures(Features):

    def __init__(self, transformer: LabelCountForSubthesauriTransformer):
        self.transformer = transformer

    def transform(self, X: LabelCalibrationData):
        rows = len(X.predicted_no_of_labels)
        no_of_predicted_labels = self.transformer.transform(
            X.predicted_labels
        )
        subthesauri_len = len(self.transformer.subthesauri_)
        data = np.zeros((rows, 2 * subthesauri_len))
        data[:, :subthesauri_len] = X.predicted_no_of_labels
        data[:, subthesauri_len:] = (
                X.predicted_no_of_labels - no_of_predicted_labels
        )
        return data
