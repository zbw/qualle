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

import logging
from collections import defaultdict
from typing import List, Set

import numpy as np
from rdflib import URIRef, Graph
from rdflib.namespace import SKOS, RDF
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

from qualle.features.label_calibration.base import AbstractLabelCalibrator, \
    AbstractLabelCalibrationFeatures
from qualle.label_calibration.category import MultiCategoryLabelCalibrator
from qualle.models import Labels, Documents, LabelCalibrationData


class LabelCountForSubthesauriTransformer(BaseEstimator, TransformerMixin):
    """Compute count of labels per Subthesauri for a given RDF Graph."""
    def __init__(
            self,
            graph: Graph,
            subthesaurus_type_uri: URIRef,
            concept_type_uri: URIRef,
            subthesauri: List[URIRef],
            concept_uri_prefix: str
    ):
        self.graph = graph
        self.subthesaurus_type_uri = subthesaurus_type_uri
        self.concept_type_uri = concept_type_uri
        self.subthesauri = subthesauri
        self.concept_uri_prefix = concept_uri_prefix

    def fit(self, X=None, y=None):
        self.mapping_ = defaultdict(lambda: [False] * len(self.subthesauri))
        self.concept_uri_prefix_len_ = len(
            self.concept_uri_prefix.rstrip('/') + '/'
        )

        for idx, s in enumerate(self.subthesauri):
            concepts = self._get_concepts_for_thesaurus(s)
            for c in concepts:
                self.mapping_[c][idx] = True
        return self

    def transform(self, X: List[Labels]) -> np.array:
        """Transform rows of concepts to 2-dimensional count array

        Each row in the result array  contains in each column the total
        amount of labels per subthesauri. The subthesauri are given
        in the same order as in the list passed to the constructor.
        """
        check_is_fitted(self)
        count_matrix = np.zeros((len(X), len(self.subthesauri)))
        for row_idx, row in enumerate(X):
            for concept in row:
                for j, is_in_subthesauri in enumerate(self.mapping_[concept]):
                    count_matrix[row_idx, j] = count_matrix[row_idx, j] + int(
                        is_in_subthesauri)
        return count_matrix

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
            regressor_class=ExtraTreesRegressor
    ):
        self.transformer = transformer
        self.regressor_class = regressor_class

    def fit(self, X: Documents, y: List[Labels]):
        self.calibrator_ = MultiCategoryLabelCalibrator(
            regressor_class=self.regressor_class
        )
        y_transformed = self.transformer.transform(y)
        self.calibrator_.fit(X, y_transformed)
        return self

    def predict(self, X: Documents):
        check_is_fitted(self)
        return self.calibrator_.predict(X)


class ThesauriLabelCalibrationFeatures(AbstractLabelCalibrationFeatures):

    def __init__(self, transformer: LabelCountForSubthesauriTransformer):
        self.transformer = transformer

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: LabelCalibrationData):
        rows = len(X.predicted_no_of_labels)
        no_of_predicted_labels = self.transformer.transform(
            X.predicted_labels
        )
        data = np.zeros((rows, 2, len(self.transformer.subthesauri)))
        data[:, 0] = X.predicted_no_of_labels
        data[:, 1] = X.predicted_no_of_labels - no_of_predicted_labels
        return data
