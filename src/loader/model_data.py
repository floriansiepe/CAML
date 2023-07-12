from dataclasses import dataclass

import pandas as pd
from darts import TimeSeries
from sklearn.base import RegressorMixin


@dataclass
class ModelData:
    model: RegressorMixin
    test_x: TimeSeries
    test_y: TimeSeries
    train_x: TimeSeries
    train_y: TimeSeries
    validation_x: TimeSeries
    validation_y: TimeSeries