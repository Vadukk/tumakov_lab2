import pandas as pd
import numpy as np

import os
from catboost import CatBoostRegressor

from src.config import *
from src.utils import *

model_path = '../../models/'
processed_path = '../../data/processed/'


def predict_values():
    df = pd.read_csv(os.path.join(processed_path, 'test.csv'))
    df.set_index('Id', inplace=True)

    model = CatBoostRegressor()

    model.load_model(os.path.join(model_path, 'model'),
                     format='cbm')
    model_features = X_cols
    res = model.predict(df[model_features])
    df.loc[:, TARGET] = res

    df[[TARGET]].to_csv(os.path.join(processed_path, 'solution.csv'))


if __name__ == '__main__':
    predict_values()

    print('Solution values have been predicted. ')
