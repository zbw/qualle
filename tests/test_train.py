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
from qualle.models import TrainData
from qualle.train import Trainer


def test_train_trains_qe_pipeline(mocker):
    train_data = TrainData(
        docs=['Title'] * 20,
        predicted_concepts=[['concept']] * 20,
        true_concepts=[['concept']] * 20
    )
    t = Trainer(train_data)
    spy = mocker.spy(t._qe_p, 'train')
    t.train()

    spy.assert_called_once()
