#  Copyright 2021-2025 ZBW – Leibniz Information Centre for Economics
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

from qualle.features.base import Features
from qualle.features.label_calibration.base import AbstractLabelCalibrator
from qualle.label_calibration.category import MultiCategoryLabelCalibrator
from qualle.models import Labels, Documents, LabelCalibrationData, Matrix
from qualle.utils import get_logger


class NotInitializedException(Exception):
    pass


class Thesaurus:
    """Handle access to RDF Graph."""

    def __init__(
        self,
        graph: Graph,
        subthesaurus_type_uri: URIRef,
        concept_type_uri: URIRef,
        concept_uri_prefix: str,
    ):
        self.graph = graph
        self.subthesaurus_type_uri = subthesaurus_type_uri
        self.concept_type_uri = concept_type_uri
        self.concept_uri_prefix_len_ = 1 + len(concept_uri_prefix.rstrip("/"))

    @lru_cache(maxsize=1000)
    def get_concepts_for_subthesaurus(self, subthesaurus: URIRef) -> Set:
        concepts = set()
        for x in self.graph[subthesaurus : SKOS.narrower]:
            if (x, RDF.type, self.subthesaurus_type_uri) in self.graph:
                concepts_from_subthesaurus = self.get_concepts_for_subthesaurus(x)
                concepts = concepts.union(concepts_from_subthesaurus)
            elif (x, RDF.type, self.concept_type_uri) in self.graph:
                concepts.add(self.extract_concept_id_from_uri_ref(x))
            else:
                logging.warning("unknown narrower type %s", str(x))
        return concepts

    def get_all_subthesauri(self) -> List[URIRef]:
        return list(self.graph[: RDF.type : self.subthesaurus_type_uri])

    def extract_concept_id_from_uri_ref(self, concept_uri: URIRef):
        return concept_uri.toPython()[self.concept_uri_prefix_len_ :]


class LabelCountForSubthesauriTransformer:
    """Compute count of labels per Subthesauri for a given RDF Graph."""

    def __init__(
        self,
        use_sparse_count_matrix: bool,
    ):
        self._use_sparse_count_matrix = use_sparse_count_matrix
        self._mapping = None
        self._logger = get_logger()
        self._subthesauri_count = 0

    def init(self, thesaurus: Thesaurus, subthesauri: Optional[List[URIRef]]):
        self._mapping = dict()
        self._subthesauri_count = len(subthesauri)

        for idx, s in enumerate(subthesauri):
            concepts = thesaurus.get_concepts_for_subthesaurus(s)
            for c in concepts:
                if c not in self._mapping:
                    self._mapping[c] = [False] * self._subthesauri_count
                self._mapping[c][idx] = True
        return self

    def transform(self, X: List[Labels]) -> Matrix:
        """Transform rows of concepts to 2-dimensional count matrix

        Each row in the result matrix contains in each column the total
        amount of labels per subthesaurus.
        """
        if not self._mapping:
            raise NotInitializedException()

        if self._use_sparse_count_matrix:
            return self._transform_with_sparse_matrix(X)
        else:
            return self._transform_without_sparse_matrix(X)

    def _transform_with_sparse_matrix(self, X: List[Labels]) -> Matrix:
        values = []
        row_inds = []
        col_inds = []
        for row_idx, row in enumerate(X):
            subthesauri_counts = self._extract_subthesauri_counts(row)
            for col_idx, count in enumerate(subthesauri_counts):
                if count:
                    values.append(count)
                    row_inds.append(row_idx)
                    col_inds.append(col_idx)

        return coo_matrix(
            (values, (row_inds, col_inds)), shape=(len(X), self._subthesauri_count)
        )

    def _transform_without_sparse_matrix(self, X: List[Labels]) -> Matrix:
        count_matrix = np.zeros((len(X), self._subthesauri_count))
        for row_idx, row in enumerate(X):
            subthesauri_counts = self._extract_subthesauri_counts(row)
            count_matrix[row_idx] = subthesauri_counts

        return count_matrix

    def _extract_subthesauri_counts(self, labels: Labels):
        subthesauri_counts = [0] * self._subthesauri_count
        for concept in labels:
            subthesauri_membership = self._mapping.get(concept)
            if not subthesauri_membership:
                self._logger.warning(
                    'Concept "%s" does not appear in any subthesaurus specified'
                    " for label calibration. "
                    "Will ignore this concept for label calibration. ",
                    concept,
                )
            else:
                for j, is_in_subthesauri in enumerate(subthesauri_membership):
                    if is_in_subthesauri:
                        subthesauri_counts[j] = subthesauri_counts[j] + 1
        return subthesauri_counts


class ThesauriLabelCalibrator(AbstractLabelCalibrator):
    def __init__(
        self,
        transformer: LabelCountForSubthesauriTransformer,
        regressor_class=ExtraTreesRegressor,
        regressor_params=None,
    ):
        self.transformer = transformer
        self.regressor_class = regressor_class
        self.regressor_params = regressor_params

    def fit(self, X: Documents, y: List[Labels]):
        self.calibrator_ = MultiCategoryLabelCalibrator(
            regressor_class=self.regressor_class, regressor_params=self.regressor_params
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
        no_of_predicted_labels = self.transformer.transform(X.predicted_labels)
        transformed_cols_len = no_of_predicted_labels.shape[1]
        data = np.zeros((rows, 2 * transformed_cols_len))
        data[:, :transformed_cols_len] = X.predicted_no_of_labels
        data[:, transformed_cols_len:] = (
            X.predicted_no_of_labels - no_of_predicted_labels
        )
        return data
