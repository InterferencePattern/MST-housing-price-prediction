import numpy as np
import pandas as pd
import re

def process_data(train_path, test_path):
    """
    Takes train and test dataset paths as arguments (./train.csv),
    performs transformations on the features, saves the processed
    dataframes as new csv files with new names.
    """

    # Load data as dataframes
    train = pd.read_csv(train_path, index_col = 0)
    test = pd.read_csv(test_path, index_col = 0)

    # Store SalePrice from train datatframe, save for later, drop SalePrice from train
    prices = train.SalePrice
    train.drop("SalePrice", axis=1, inplace=True)

    # Merge training and test dataframes
    df = pd.concat([train, test])

    # Add '_' to the beginning of feature names if they start with a number
    df.columns = list(map(lambda x: '_' + x if re.match('^\d', x) else x, df.columns))

    # Drop all unused columns
    drop_cols = ['Street', 'Alley', 'Utilities', 'LandSlope', 'Condition2', 'YearRemodAdd', 'RoofMatl',
                'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtHalfBath', 'GarageQual',
                'GarageCond', 'PoolQC', 'MiscFeature', 'MiscVal', 'YrSold']
    df.drop(columns = drop_cols, inplace=True)

    # Transformations: Box-Cox, imputation, etc...
    # features to boxcox/impute/etc = ['LotFrontage', 'LotArea', 'OverallQual', 'BsmtFinSF1'?,
    #                                'BsmtUnfSF'?, 'TotalBsmtSF'?, 'TotRmsAbvGrd', 'GarageYrBlt']

    # Transformations: Category/feature merging/creation
    # features to merge/create = ['LotConfig', 'Condition1'?, 'OverallCond', 'YearBuilt', 'Exterior1st',
    #                            'Electrical', '_1stFlrSF', '_2ndFlrSF', 'FullBath', 'BedroomAbvGr',
    #                            'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'SaleType', 'SaleCondition']

    # Transformations: Boolean feature encoding
    # boolean features = ['LotShape', 'Exterior2nd', 'CentralAir', 'BsmtFullBath', 'HalfBath',
    #                   'Functional', 'WoodDeckSF'?, 'OpenPorchSF'?, 'EnclosedPorchSF'?, '_3SsnPorch'?,
    #                   'ScreenPorch'?, 'PoolArea']

    # Transformations: Ordinal feature encoding
    # ordinal features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    #                    'HeatingQC', 'FireplaceQu', 'GarageFinish']

    # Transformations: Dummify feature encoding
    # dummify features = ['MSSubClass', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
    #                   'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st',
    #                   'MasVnrType', 'Foundation', 'BsmtFinType1', 'Electrical', '_2ndFlrSF',
    #                   'BedroomAbvGr', 'KitchenAbvGr', 'GarageType', 'GarageCars', 'PavedDrive',
    #                   'Fence', 'MoSold', 'SaleType', 'SaleCondition']

    # Split dataframe into test and train by train length
    final_train = df.iloc[0:len(train),:]
    final_test = df.iloc[len(train):,:]

    # Add SalePrice column back to train dataframe
    final_train = pd.concat([final_train, prices], axis = 1)

    # Save dataframes to './ProcessedTrain.csv' and './ProcessedTest.csv'
    final_train.to_csv('./ProcessedTrain.csv')
    final_test.to_csv('./ProcessedTest.csv')
