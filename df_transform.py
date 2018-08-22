import numpy as np
import pandas as pd

def transform(train_path, test_path):
    """ Takes train and test dataset paths as arguments, saves the tranformed data sets
    as csv files in the data folder under new names """

    train = pd.read_csv(train_path, index_col = 0)
    test = pd.read_csv(test_path, index_col = 0)

    # Concat dataframes Here

    # Drop all unused columns Here

    # Transformations

    # Split dataframe by index

    # Save dataframes to './data/train_transform.csv' and './data/test_transform.csv'
