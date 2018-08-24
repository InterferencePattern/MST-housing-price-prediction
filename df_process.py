import numpy as np
import pandas as pd
import re
from scipy.special import boxcox1p

def process_data(train_path, test_path, train_opt_path='p_train.csv', test_opt_path='p_test.csv', price_opt_path='prices.csv'):
    """
    Takes train and test dataset paths as arguments (./train.csv),
    performs transformations on the features, saves the processed
    dataframes as new csv files with new names.
    """

    # Load data as dataframes
    train = pd.read_csv(train_path, index_col = 0)
    test = pd.read_csv(test_path, index_col = 0)

    # Drop two outliers with very high GrLivArea and low SalePrice
    train = train.drop(train[(train.GrLivArea > 4000) & (train.SalePrice < 300000)].index)

    # Drop an outlier in LotFrontage
    train = train.drop(train[train.LotFrontage > 300].index)

    # Store SalePrice from train datatframe as log(SalePrice), save for later, drop SalePrice from train
    saleprice = np.log(train.SalePrice)
    train.drop("SalePrice", axis=1, inplace=True)

    # Combine training and test dataframes
    df = pd.concat([train, test])

    # Add '_' to the beginning of feature names if they start with a number
    df.columns = list(map(lambda x: '_' + x if re.match('^\d', x) else x, df.columns))

    # Drop features that do not appear to give important information
    drop_cols = ['Street', 'Alley', 'Utilities', 'LandSlope', 'Condition2', 'YearRemodAdd', 'RoofMatl',
                'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'LowQualFinSF', 'BsmtHalfBath', 'GarageQual',
                'GarageCond', 'PoolQC', 'MiscFeature', 'MiscVal', 'YrSold']
    df.drop(columns = drop_cols, inplace=True)

    ###################
    ### NA Handling ###
    ###################

    # Fence : Data description says NA means "no fence."
    df.Fence = df.Fence.fillna("None")

	# FireplaceQu : Data description says NA means "no fireplace."
    df.FireplaceQu = df.FireplaceQu.fillna("None")

	# LotFrontage : Because LotFrontage will most likely be similar within neighborhoods, we can replace these NAs
    # with the median LotFrontage for each neighborhood.
	# Group by neighborhood and replace NAs with the median LotFrontage for each neighborhood
    df.LotFrontage = df.groupby("Neighborhood").LotFrontage.transform(lambda x: x.fillna(x.median())).astype(int)

	# GarageType and GarageFinish: Replace NAs with 'None'.
    df.GarageType = df.GarageType.fillna('None')
    df.GarageFinish = df.GarageFinish.fillna('None')

    # GarageYrBlt : Replace NAs in GarageYrBlt with value in YearBuilt, convert to int
    df.GarageYrBlt = df.GarageYrBlt.fillna(df.YearBuilt).astype(int)

	# GarageArea and GarageCars : Replacing NAs with 0 (Since No garage = no cars), convert to int
    df.GarageArea = df.GarageArea.fillna(0).astype(int)
    df.GarageCars = df.GarageCars.fillna(0).astype(int)

	# _2ndFlrSF, BsmtFinSF1, BsmtUnfSF and TotalBsmtSF : Replace NAs with 0 since missing values are likely due to no basement
    # or no second floor.
    for col in ['_2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF']:
        df[col] = df[col].fillna(0)

	# BsmtQual, BsmtCond, BsmtExposure and BsmtFinType1 : NAs mean there is no basement, replace with 'None'.
    for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1']:
        df[col] = df[col].fillna('None')

	# MasVnrArea and MasVnrType : NAs most likely means no masonry veneer for these houses. Fill 0 for the area and 'None' for the type.
    df.MasVnrType = df.MasVnrType.fillna("None")
    df.MasVnrArea = df.MasVnrArea.fillna(0)

	# MSZoning (general zoning classification) : 'RL' is by far the most common value. We can fill in NAs with 'RL'.
    df.MSZoning = df.MSZoning.fillna(df.MSZoning.mode()[0])

	# Functional : Data description says NA means Typical.
    df.Functional = df.Functional.fillna("Typ")

	# Electrical : Only one NA value; we can replace NA with the mode of this feature which is 'SBrkr'.
    df.Electrical = df.Electrical.fillna(df.Electrical.mode()[0])

	# KitchenQual: Only one NA value; we can replace NA with the mode of this feature which is 'TA'.
    df.KitchenQual = df.KitchenQual.fillna(df.KitchenQual.mode()[0])

	# Exterior1st and Exterior2nd : Both Exterior 1 & 2 only have one NA; we can replace it with the mode.
    df.Exterior1st = df.Exterior1st.fillna(df.Exterior1st.mode()[0])
    df.Exterior2nd = df.Exterior2nd.fillna(df.Exterior2nd.mode()[0])

	# SaleType : Fill NA with mode which is "WD".
    df.SaleType = df.SaleType.fillna(df.SaleType.mode()[0])

    ################################
    ### Transformations: Box-Cox ###
    ################################
    # Features to transform with boxcox = ['LotArea', 'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF',
    #                                     'TotalBsmtSF', 'TotRmsAbvGrd']

    # LotArea : Fanning on upper end
    best_lambda = -.127
    df.LotArea = boxcox1p(df.LotArea, best_lambda)

    # OverallQual : Some fanning
    best_lambda = .7
    df.OverallQual = boxcox1p(df.OverallQual, best_lambda)

    # BsmtFinSF1 : Lots of 0s
    best_lambda = .168
    df.BsmtFinSF1 = boxcox1p(df.BsmtFinSF1, lam)

    # BsmtUnfSF : Many 0s, MIGHT BE BETTER TO NOT TRANSFORM
    best_lambda = .208
    df.BsmtUnfSF = boxcox1p(df.BsmtUnfSF, best_lambda)

    # TotalBsmtSF : Many 0s, TRANSFORMATION QUESTIONABLE
    best_lambda = .595
    df.TotalBsmtSF = boxcox1p(df.TotalBsmtSF, best_lambda)

    # TotRmsAbvGrd : Small football effect
    best_lambda = -.138
    df.TotRmsAbvGrd = boxcox1p(df.TotRmsAbvGrd, best_lambda)

    ####################################################
    # Transformations: Category/feature merging/creation
    ####################################################
    # Features to combine/create = ['LotConfig', 'Condition1', 'OverallCond', 'YearBuilt', 'Exterior1st',
    #                            '_1stFlrSF', '_2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
    #                            'GarageCars', 'SaleType', 'SaleCondition']

    # LotConfig : Combine FR2 and FR3 categories
    df.LotConfig = df.LotConfig.apply(lambda x: 'FR2' if x == 'FR3' else x)

    # Condition1 : Combine railroad-adjacent categories to one RR category
    railroad = ['RRAn', 'RRAe', 'RRNn', 'RRNe']
    df.Condition1 = df.Condition1.apply(lambda x: 'NearRR' if x in railroad else x)

    # OverallCond : Reassign all values greater than 5 to 5
    df.OverallCond = df.OverallCond.apply(lambda x: 5 if x > 5 else x)

    # YearBuilt : Add YearBuilt**2 feature
    df['YearBuiltSqr'] = df.YearBuilt**2

    # Exterior1st : Combine categories that have less than 20 observations into an 'Other' category
    # Creates dictionary with count of observations in each category
    lumped = dict(df.groupby("Exterior1st").size())
    df.Exterior1st = df.Exterior1st.apply(lambda x: 'Other' if lumped[x] < 20 else x)

    # _1stFlrSF and _2ndFlrSF : Create new feature TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF,
    # combine 1st and 2nd into _1stFlrSF, convert _2ndFlrSF to boolean (_2ndFlrSF > 0), rename
    # _2ndFlrSF to Has2ndFlr
    df["TotalSF"] = (df._1stFlrSF + df._2ndFlrSF + df.TotalBsmtSF).astype(int)
    df._2ndFlrSF = df._2ndFlrSF.apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={'_2ndFlrSF':'Has2ndFlr'}, inplace=True)

    # FullBath : Combine 0 and 1 to a 0-1 category and convert to string for dummification
    df.FullBath = df.FullBath.apply(lambda x: 1 if x == 0 else x).astype(str)

    # BedroomAbvGr : Combine 0 and 1 to 0-1, reassign greater than 5 to 5, convert to string for dummification
    df.BedroomAbvGr = df.BedroomAbvGr.apply(lambda x: 1 if x == 0 else (5 if x > 5 else x)).astype(str)

    # KitchenAbvGr : Combine 0 and 1 to 0-1, reassign greater than 2 to 2, convert to string for dummification
    df.KitchenAbvGr = df.KitchenAbvGr.apply(lambda x: 1 if x == 0 else (2 if x > 2 else x)).astype(str)

    # GarageCars : Reassign all values greater than 3 to 3, convert to string for dummification
    df.GarageCars = df.GarageCars.apply(lambda x: 3 if x > 3 else x).astype(str)

    # SaleType : Combine other than 'WD' and 'New' into new category 'Other'
    df.SaleType = df.SaleType.apply(lambda x: 'Other' if x not in ['WD', 'New'] else x)

    # SaleCondition : Combine other than 'Abnorml' and 'Partial' into 'Normal'
    df.SaleCondition = df.SaleCondition.apply(lambda x: 'Normal' if x not in ['Abnorml', 'Partial', 'Normal'] else x)

    #################################################
    ### Transformations: Boolean feature encoding ###
    #################################################
    # Boolean features = ['LotShape', 'CentralAir', 'Electrical', 'BsmtFullBath', 'HalfBath',
    #                   'Fireplaces', 'Functional', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    #                   '_3SsnPorch', 'ScreenPorch', 'PoolArea']

    # LotShape : Convert to 1 if 'Reg', else 0
    df.LotShape = df.LotShape.apply(lambda x: 1 if x == 'Reg' else 0)

    # CentralAir :
    df.CentralAir = df.CentralAir.apply(lambda x: 1 if x == "Y" else 0)

    # Electrical : Convert to 1 if 'Sbrkr', else 0
    df.Electrical = df.Electrical.apply(lambda x: 1 if x ==" Sbrkr" else 0)

    # BsmtFullBath : Convert to 1 if > 0, else 0, rename feature to 'HasBsmtFullBath'
    df.BsmtFullBath = df.BsmtFullBath.apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={'BsmtFullBath':'HasBsmtFullBath'}, inplace=True)

    # HalfBath : Convert to 1 if > 0, else 0, rename feature 'HasHalfBath'
    df.HalfBath = df.HalfBath.apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={'HalfBath':'HasHalfBath'}, inplace=True)

    # Fireplaces : Convert to 1 if > 0, else 0, rename feature to 'HasFireplaces'
    df.Fireplaces = df.Fireplaces.apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={'Fireplaces':'HasFireplaces'}, inplace=True)

    # Functional : Convert to 1 if 'Typ', else 0
    df.Functional = df.Functional.apply(lambda x: 1 if x == 'Typ' else 0)

    # WoodDeckSF, OpenPorchSF, EnclosedPorch, _3SsnPorch and ScreenPorch : Convert to 1 if > 0, else 0, rename features
    for col in ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '_3SsnPorch', 'ScreenPorch']:
        df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
    porch_names = {'WoodDeckSF': 'HasWoodDeck', 'OpenPorchSF': 'HasOpenPorch', 'EnclosedPorch': 'HasEnclosedPorch',
                  '_3SsnPorch': 'Has3SsnPorch', 'ScreenPorch': 'HasScreenPorch'}
    df.rename(columns = porch_names, inplace=True)

    # PoolArea : Convert to 1 if > 0, else 0, rename feature to 'HasPool'
    df.PoolArea = df.PoolArea.apply(lambda x: 1 if x > 0 else 0)
    df.rename(columns={'PoolArea': 'HasPool'}, inplace=True)

    #################################################
    ### Transformations: Ordinal feature encoding ###
    #################################################
    # Ordinal features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    #                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish']

    # ExterQual : Convert to ordinal values {Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df.ExterQual = df.ExterQual.replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    # ExterCond : Convert to ordinal values {'Po': 1, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    # Very few 'Po's, merge with 'Fa'
    df.ExterCond = df.ExterCond.replace({'Po': 1, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    # BsmtQual : Convert to ordinal values {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df.BsmtQual = df.BsmtQual.replace({'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    # BsmtCond : Convert to ordinal values {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}
    df.BsmtCond = df.BsmtCond.replace({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4})

    # BsmtExposure : Convert to ordinal values {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    df.BsmtExposure = df.BsmtExposure.replace({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})

    # HeatingQC : Convert to ordinal values {Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df.HeatingQC = df.HeatingQC.replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    # KitchenQual : Convert to ordinal values {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df.KitchenQual = df.KitchenQual.replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    # FireplaceQu : Convert to ordinal values {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df.FireplaceQu = df.FireplaceQu.replace({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    # GarageFinish : Convert to ordinal values {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    df.GarageFinish = df.GarageFinish.replace({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})

    #################################################
    ### Transformations: Dummify feature encoding ###
    #################################################
    # Dummify features = ['MSSubClass', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
    #                   'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st',
    #                   'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'FullBath',
    #                   'BedroomAbvGr', 'KitchenAbvGr', 'GarageType', 'GarageCars', 'PavedDrive',
    #                   'Fence', 'MoSold', 'SaleType', 'SaleCondition']

    # Convert numeric features to string for dummifcation
    df.MSSubClass = df.MSSubClass.astype(str)
    df.MoSold = df.MoSold.astype(str)

    df = pd.get_dummies(df, drop_first=True)

    #######################################
    ### Split Dataframe and Save to CSV ###
    #######################################

    # Split dataframe into test and train by train length
    final_train = df.iloc[0:len(train),:]
    final_test = df.iloc[len(train):,:]

    # Save dataframes to csv with file names 'train_opt_path' and 'test_opt_path'
    final_train.to_csv(train_opt_path)
    final_test.to_csv(test_opt_path)
    saleprice.to_csv(price_opt_path)
