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
from typing import Set, Type, List

from sklearn.base import RegressorMixin

from qualle.features.base import Features
from qualle.features.combined import CombinedFeaturesData, CombinedFeatures
from qualle.features.confidence import ConfidenceFeatures
from qualle.features.label_calibration.base import AbstractLabelCalibrator
from qualle.features.label_calibration.simple_label_calibration import (
    SimpleLabelCalibrationFeatures,
)
from qualle.features.label_calibration.thesauri_label_calibration import (
    ThesauriLabelCalibrationFeatures,
)
from qualle.features.text import TextFeatures
from qualle.models import PredictData, LabelCalibrationData
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import QualityEstimator

FeaturesTypes = Set[Type[Features]]


class Trainer:
    def __init__(
        self,
        train_data,
        label_calibrator: AbstractLabelCalibrator,
        quality_regressor: RegressorMixin,
        features: List[Features],
        should_debug=False,
    ):
        combined_features = CombinedFeatures(features)
        rp = QualityEstimator(regressor=quality_regressor, features=combined_features)
        self._qe_p = QualityEstimationPipeline(
            recall_predictor=rp,
            label_calibrator=label_calibrator,
            should_debug=should_debug,
            features_data_mapper=FeaturesDataMapper(set(f.__class__ for f in features)),
        )
        self._train_data = train_data

    def train(self) -> QualityEstimationPipeline:
        self._qe_p.train(self._train_data)

        return self._qe_p


class FeaturesDataMapper:
    def __init__(self, features_types: FeaturesTypes):
        self._features_types = features_types

    def __call__(
        self, p_data: PredictData, l_data: LabelCalibrationData
    ) -> CombinedFeaturesData:
        features_data = dict()
        for ftype in self._features_types:
            if ftype == ConfidenceFeatures:
                features_data[ConfidenceFeatures] = p_data.scores
            if ftype == TextFeatures:
                features_data[TextFeatures] = p_data.docs
            if ftype in (
                SimpleLabelCalibrationFeatures,
                ThesauriLabelCalibrationFeatures,
            ):
                features_data[ftype] = l_data
        return features_data
