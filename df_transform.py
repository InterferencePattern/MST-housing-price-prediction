import numpy as np
import pandas as pd
import re
import math

def transform(train_path, test_path, train_opt_path, test_opt_path):
    """ Takes train and test dataset paths as arguments, saves the tranformed data sets
    as csv files in the data folder under new names """

    train = pd.read_csv(train_path, index_col = 0)
    test = pd.read_csv(test_path, index_col = 0)

    # Save SalePrice for later
    saleprice = pd.DataFrame({"SalePrice":train.SalePrice})

    # Drop SalePrice from training set
    train.drop("SalePrice", axis=1, inplace=True)

    # Concat training and test dataframes
    df = pd.concat([train, test], sort=False)

    # Rename columns that are incompatible with Pandas (periods, etc)
    col_names = [key for key in dict(df.dtypes) if re.search("^[^A-Z]",key) != None]
    new_col_names = ['FirstFlrSF', 'SecondFlrSF', 'ThirdSsnPorch']
    new_col_names = dict(zip(col_names,new_col_names))
    df.rename(columns=new_col_names, inplace=True)

    # Drop all unused columns Here
    df.drop('Street',axis=1, inplace=True)
    df.drop('Alley',axis=1, inplace=True)
    df.drop('Utilities',axis=1, inplace=True)
    df.drop('LandSlope',axis=1, inplace=True)
    df.drop('Condition2',axis=1, inplace=True)
    df.drop('YearRemodAdd',axis=1, inplace=True)
    df.drop('RoofMatl',axis=1, inplace=True)
    df.drop('BsmtFinType2',axis=1, inplace=True)
    df.drop('BsmtFinSF2',axis=1, inplace=True)
    df.drop('Heating',axis=1, inplace=True)
    df.drop('LowQualFinSF',axis=1, inplace=True)
    df.drop('BsmtHalfBath',axis=1, inplace=True)
    df.drop('GarageQual',axis=1, inplace=True)
    df.drop('GarageCond',axis=1, inplace=True)
    df.drop('PoolQC',axis=1, inplace=True)
    df.drop('MiscFeature',axis=1, inplace=True)
    df.drop('MiscVal',axis=1, inplace=True)
    df.drop('YrSold',axis=1, inplace=True)

    ## Transformations
    # CentralAir - boolean
    df.CentralAir = df.CentralAir.apply(lambda x: True if x=="Y" else False)

    # Electrical - split into Sbrkr and NotSbrkr
    df.Electrical.fillna("Sbrkr", inplace=True)
    df.Electrical = df.Electrical.apply(lambda x: "Sbrkr" if x=="Sbrkr" else "NotSbrkr")

    # FirstFlrSF, SecondFlrSF - combine 1st and 2nd floor SF; have a second param for "has 2 floors" (2ndSF>0)
    df.SecondFlrSF.fillna(0,inplace=True)
    df["TotalSF"] = df.FirstFlrSF + df.SecondFlrSF
    df["HasTwoFloors"] = df.SecondFlrSF.apply(lambda x: True if x>0 else False)
    df.drop('FirstFlrSF', axis=1, inplace=True)
    df.drop('SecondFlrSF', axis=1, inplace=True)

    # BsmtFullBath - boolean ("HasBsmtFullBath" T for >0, F for ==0)
    df.BsmtFullBath.fillna(0, inplace=True)
    df.BsmtFullBath = df.BsmtFullBath.apply(lambda x: True if x>0 else False)

    # FullBath - 0-1-->1, 2-->2, 3-->3
    df.FullBath = df.FullBath.apply(lambda x: 'one_or_fewer_full_baths' if x<=1 else 'two_full_baths' if x==2 else 'three_or_more_full_baths')

    # HalfBath - boolean ("HasHalfBath" >0, T, ==0 F)
    df.HalfBath = df.HalfBath.apply(lambda x: True if x>0 else False)

    # BedroomAbvGr - categorical: 0-1, 2, 3, 4, 5+
    df.BedroomAbvGr = df.BedroomAbvGr.apply(lambda x: 'bedAbvGr_one_or_fewer' if x <=1 else 'bedAbvGr_two' if x==2 else 'bedAbvGr_three' if x==3 else 'bedAbvGr_four' if x == 4 else 'bedAbvGr_five_or_more')

    # KitchenAbvGr - categorical: 0-1, 2+
    df.KitchenAbvGr = df.KitchenAbvGr.apply(lambda x: 'KitchenAbvGr_one_or_fewer' if x <=1 else 'KitchenAbvGr_two_or_more')

    # TotRmsAbvGrd - log (maybe box-cox)
    df.TotRmsAbvGrd = df.TotRmsAbvGrd.apply(lambda x: math.log(x))

    # Functional - Typ --> T, ~Typ --> F
    df.Functional = df.Functional.apply(lambda x: 'Typ' if x=='Typ' else 'NotTyp')

    # Fireplaces - 0, 1, >=2
    df.Fireplaces = df.Fireplaces.apply(lambda x: 'Fireplaces_zero' if x==0 else 'Fireplaces_one' if x==1 else 'Fireplaces_two_or_more')

    # FireplaceQu - {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    df.FireplaceQu.fillna('NA',inplace=True)
    fireplace = {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    df.FireplaceQu = df.FireplaceQu.apply(lambda x: fireplace[x])

    # GarageYrBlt - numeric (replace NA with YearBuilt)
    df.GarageYrBlt.fillna(df.YearBuilt, inplace=True)

    # GarageFinish - {'NA':0,'Unf':1,'RFn':2,'Fin':3}
    df.GarageFinish.fillna('NA', inplace=True)
    garagefinish = {'NA':0,'Unf':1,'RFn':2,'Fin':3}
    df.GarageFinish = df.GarageFinish.apply(lambda x: garagefinish[x])

    # Encoding
    df = pd.get_dummies(df, drop_first=True, dummy_na=True)

    # Split dataframe into test and train by index
    df_finalTrain = df.iloc[0:train.shape[0],:]
    df_finalTrain = pd.concat([df_finalTrain, saleprice], axis=1, sort=False)
    df_finalTest = df.iloc[train.shape[0]:,:]

    # Export
    df_finalTrain.to_csv(train_opt_path)
    df_finalTest.to_csv(test_opt_path)
