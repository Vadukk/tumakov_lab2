import pandas as pd
import os

from src.features.build_features import create_complete_dataset

raw_path = '../../data/raw/'
processed_path = '../../data/processed/'

if __name__ == '__main__':

    train = pd.read_csv(os.path.join(raw_path, 'train.csv'))
    test = pd.read_csv(os.path.join(raw_path, 'test.csv'))

    train_features = create_complete_dataset(train, True)
    test_features = create_complete_dataset(test, False)

    train_features.to_csv(os.path.join(processed_path, 'train.csv'), index=False)
    test_features.to_csv(os.path.join(processed_path, 'test.csv'), index=False)

    print('All data has been processed. ')
