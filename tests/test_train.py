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
from sklearn.ensemble import ExtraTreesRegressor

from qualle.features.label_calibration.simple_label_calibration import \
    SimpleLabelCalibrator, SimpleLabelCalibrationFeatures
from qualle.quality_estimation import RecallPredictor
from qualle.train import Trainer


def test_train_trains_qe_pipeline(train_data, mocker):
    t = Trainer(
        train_data=train_data,
        label_calibrator=SimpleLabelCalibrator(ExtraTreesRegressor()),
        recall_predictor=RecallPredictor(
            regressor=ExtraTreesRegressor(),
            features=SimpleLabelCalibrationFeatures()
        )
    )
    spy = mocker.spy(t._qe_p, 'train')
    t.train()

    spy.assert_called_once()
