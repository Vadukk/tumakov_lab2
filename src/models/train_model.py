import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor

from src.config import *
from src.utils import *

model_path = '../../models/'


def train_models():
    df = pd.read_csv('../../data/processed/train.csv')

    model_features = X_cols
    model = CatBoostRegressor(**model_params)

    model.fit(df[model_features], df[TARGET])

    model.save_model(os.path.join(model_path, 'model'),
                     format="cbm",
                     export_parameters=None,
                     pool=None)


if __name__ == '__main__':
    train_models()
    print('Model has been trained. ')