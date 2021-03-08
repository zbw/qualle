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
import numpy as np


class RecallPredictor:

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        data = np.zeros((len(X), 2))
        data[:, 0] = X['label_calibration']
        data[:, 1] = X['label_calibration'] - X['no_of_pred_labels']
        self.estimator.fit(data, y)

    def predict(self, X):
        data = np.zeros((len(X), 2))
        data[:, 0] = X['label_calibration']
        data[:, 1] = X['label_calibration'] - X['no_of_pred_labels']
        return self.estimator.predict(X)
