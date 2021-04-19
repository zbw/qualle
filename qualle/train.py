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
from typing import Set, Type, List

from sklearn.base import RegressorMixin

from qualle.features.base import Features
from qualle.features.combined import CombinedFeaturesData, CombinedFeatures
from qualle.features.confidence import ConfidenceFeatures
from qualle.features.label_calibration.base import AbstractLabelCalibrator
from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrationFeatures
from qualle.features.label_calibration.thesauri_label_calibration import \
    ThesauriLabelCalibrationFeatures
from qualle.models import PredictData, LabelCalibrationData
from qualle.pipeline import QualityEstimationPipeline
from qualle.quality_estimation import RecallPredictor

FeaturesTypes = Set[Type[Features]]


class Trainer:

    def __init__(
            self, train_data,
            label_calibrator: AbstractLabelCalibrator,
            recall_predictor_regressor: RegressorMixin,
            features: List[Features],
            should_debug=False
    ):
        combined_features = CombinedFeatures(features)
        rp = RecallPredictor(
            regressor=recall_predictor_regressor, features=combined_features
        )
        self._qe_p = QualityEstimationPipeline(
            recall_predictor=rp,
            label_calibrator=label_calibrator,
            should_debug=should_debug,
            features_data_mapper=FeaturesDataMapper(
                set(f.__class__ for f in features)
            )
        )
        self._train_data = train_data

    def train(self) -> QualityEstimationPipeline:
        self._qe_p.train(self._train_data)

        return self._qe_p


class FeaturesDataMapper:

    def __init__(self, features_types: FeaturesTypes):
        self._features_types = features_types

    def __call__(self, p_data: PredictData, l_data: LabelCalibrationData)\
            -> CombinedFeaturesData:
        features_data = dict()
        for ftype in self._features_types:
            if ftype == ConfidenceFeatures:
                features_data[ConfidenceFeatures] = p_data.scores
            if ftype in (
                    SimpleLabelCalibrationFeatures,
                    ThesauriLabelCalibrationFeatures
            ):
                features_data[ftype] = l_data
        return features_data
