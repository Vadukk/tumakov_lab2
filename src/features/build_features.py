import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

from src.config import *

encoders_path = '../../models/Encoders/'


def create_complete_dataset(df, train_df=False):
    df = df.fillna('No')

    if train_df:
        for col in X_cols:
            if col in categorial_cols:
                encoder = LabelEncoder()

                df[col] = encoder.fit_transform(df[col])

                np.save(os.path.join(encoders_path, f'{col}_encoder.npy'), encoder.classes_)

    else:
        for col in X_cols:
            if col in categorial_cols:
                encoder = LabelEncoder()
                encoder.classes_ = np.load(os.path.join(encoders_path, f'{col}_encoder.npy'),
                                           allow_pickle=True)

                df[col] = df[col].map(dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))).fillna(999)


    #df.drop(columns='Id', inplace=True)
    for col in X_cols:
        if 'No' in list(df[col]):
            df.loc[df[col] == 'No', col] = 0

    return df




