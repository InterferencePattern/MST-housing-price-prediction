# Run 5 different kinds of models on the transformed data
# For each kind of model (ex: Linear Regression), 6 models are
# generated and their predictions, along with the "actual" values
# (when avialable) are returned in a CSV file

# For each kind of models, tuned hyperparameters are supplied as pickle
# files.  Note that the names for these are hard coded in.  Make sure
# that your pickles are in the correct directory and correctly named

# Arguments
# 1 - training data CSV - no sale price column
# 2 - training prices   - logs of prices for train ids
# 3 - test data CSV.    - no prices

# Outputs - 6 CSV files: linear_results.csv, rf1_results.csv,
# rf2_results.csv, xgb_results.csv, gb_results.csv
# Note that the names for these are hard coded (and they
# will appear in this directory)

##### Libraries #####
import sys
import pandas as pd
import numpy as np
import pickle

# ML
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

##### Load the data #####
# Training data - has a saleprice column
train = pd.read_csv(sys.argv[1])
train.set_index('Id', inplace=True)

train_prices = pd.read_csv(sys.argv[2])
train_prices.set_index('Id', inplace=True)

# Test data - doesn't have a saleprice column
to_guess = pd.read_csv(sys.argv[3])
to_guess.set_index('Id', inplace=True)

##### Open all of the pickles to get the params #####
# Linear
try:
    with open('linear_params.pkl', 'rb') as f:
        param_dict_linear = pickle.load(f)
    model_linear = LinearRegression(**param_dict_linear)
except:
    print('Warning: Running Linear Regression with default hyperparameters')
    model_linear = LinearRegression()

# RF1
try:
    with open('rf1_params.pkl', 'rb') as f:
        param_dict_rf1 = pickle.load(f)

        model_rf1 = RandomForestRegressor(**param_dict_rf1)
except:
    print('Warning: Running Random Forest 1 with default hyperparameters')
    model_rf1 = RandomForestRegressor()


# RF2
try:
    with open('rf2_params.pkl', 'rb') as f:
        param_dict_rf2 = pickle.load(f)

    model_rf2 = RandomForestRegressor(**param_dict_rf2)
except:
    print('Warning: Running Random Forest 2 with default hyperparameters')
    model_rf2 = RandomForestRegressor()


# XGBoost
try:
    with open('xgb_params.pkl', 'rb') as f:
        param_dict_xgb = pickle.load(f)

    model_xgb = XGBRegressor(**param_dict_xgb)
except:
    print('Warning: Running XGBoost with default hyperparameters')
    model_xgb = XGBRegressor()

# GradientBoosting
try:
    with open('gb_params.pkl', 'rb') as f:
        param_dict_gb = pickle.load(f)

    model_gb = GradientBoostingRegressor(**param_dict_gb)
except:
    print('Warning: Running Gradient Boosting with default hyperparameters')
    model_gb = GradientBoostingRegressor()

##### Split the training set into 5 folds #####
kf = KFold(n_splits=5)

# keep the indicies of each train and test set in a nested array
train_indicies = []
test_indicies = []
for train_index, test_index in kf.split(train):
    train_indicies = train_indicies + [train_index]
    test_indicies = test_indicies + [test_index]

# train.iloc[train_indicies[0]]             <-- first 80% train features
# train_prices.iloc[train_indicies[0]]      <-- first 80% train labels
# train.iloc[test_indicies[0]].shape        <-- first 20% test features
# train_prices.iloc[test_indicies[0]].shape <-- first 20% test labels

###### Linear Regression #####
# Data frame to store results
linear_results = pd.DataFrame({"Id":train_prices.index, "actual":train_prices.log_SalePrice})
linear_results.set_index('Id',inplace=True)

# iterate through the 5 folds to create the first 5 models and gen results
for i in range(0,5):
    model_linear.fit(train.iloc[train_indicies[i]], train_prices.iloc[train_indicies[i]])  # model on 80%
    pred = model_linear.predict(train.iloc[test_indicies[i]])                              # predict on 20%
    pred = [i[0] for i in pred]
    model_name = "model" + str(i+1)
    temp_df = pd.DataFrame({'Id':test_indicies[i], model_name:pred})    # temp DF with ID and new model results
    temp_df.set_index('Id', inplace=True)
    linear_results = linear_results.merge(temp_df, on='Id', how='left') # add to linear_results DF

# create a 6th model - train on all training, predict on all test ('to_guess')
model_linear.fit(train, train_prices)
pred = model_linear.predict(to_guess)
pred = [i[0] for i in pred]
temp_df = pd.DataFrame({'Id':to_guess.index, "model6":pred})
temp_df.set_index('Id', inplace=True)
linear_results = linear_results.merge(temp_df, on='Id', how='outer') # add to linear_results DF

# reorder cols (actual at the end)
#linear_results = linear_results[['model1','model2','model3','model4','model5','model6','actual']]
linear_results['predictions'] = linear_results[['model1','model2','model3','model4','model5','model6']].mean(axis = 1)

# output to a csv
linear_results.to_csv('linear_results.csv')

###### Random Forest 1 #####
# Data frame to store results
rf1_results = pd.DataFrame({"Id":train_prices.index, "actual":train_prices.log_SalePrice})
rf1_results.set_index('Id',inplace=True)

# iterate through the 5 folds to create the first 5 models and gen results
for i in range(0,5):
    model_rf1.fit(train.iloc[train_indicies[i]], train_prices.iloc[train_indicies[i]].log_SalePrice)  # model on 80%
    pred = model_rf1.predict(train.iloc[test_indicies[i]])                              # predict on 20%
    pred = [i for i in pred]
    model_name = "model" + str(i+1)
    temp_df = pd.DataFrame({'Id':test_indicies[i], model_name:pred})    # temp DF with ID and new model results
    temp_df.set_index('Id', inplace=True)
    rf1_results = rf1_results.merge(temp_df, on='Id', how='left') # add to rf1_results DF

# create a 6th model - train on all training, predict on all test ('to_guess')
model_rf1.fit(train, train_prices.log_SalePrice)
pred = model_rf1.predict(to_guess)
pred = [i for i in pred]
temp_df = pd.DataFrame({'Id':to_guess.index, "model6":pred})
temp_df.set_index('Id', inplace=True)
rf1_results = rf1_results.merge(temp_df, on='Id', how='outer') # add to rf1_results DF

# # reorder cols (actual at the end)
#rf1_results = rf1_results[['model1','model2','model3','model4','model5','model6','actual']]
rf1_results['predictions'] = rf1_results[['model1','model2','model3','model4','model5','model6']].mean(axis = 1)

# # output to a csv
rf1_results.to_csv('rf1_results.csv')

##### Random Forest 2 #####
# Data frame to store results
rf2_results = pd.DataFrame({"Id":train_prices.index, "actual":train_prices.log_SalePrice})
rf2_results.set_index('Id',inplace=True)

# iterate through the 5 folds to create the first 5 models and gen results
for i in range(0,5):
    model_rf2.fit(train.iloc[train_indicies[i]], train_prices.iloc[train_indicies[i]].log_SalePrice)  # model on 80%
    pred = model_rf2.predict(train.iloc[test_indicies[i]])                              # predict on 20%
    pred = [i for i in pred]
    model_name = "model" + str(i+1)
    temp_df = pd.DataFrame({'Id':test_indicies[i], model_name:pred})    # temp DF with ID and new model results
    temp_df.set_index('Id', inplace=True)
    rf2_results = rf2_results.merge(temp_df, on='Id', how='left') # add to rf2_results DF

# create a 6th model - train on all training, predict on all test ('to_guess')
model_rf2.fit(train, train_prices.log_SalePrice)
pred = model_rf2.predict(to_guess)
pred = [i for i in pred]
temp_df = pd.DataFrame({'Id':to_guess.index, "model6":pred})
temp_df.set_index('Id', inplace=True)
rf2_results = rf2_results.merge(temp_df, on='Id', how='outer') # add to rf2_results DF

# # reorder cols (actual at the end)
#rf2_results = rf2_results[['model1','model2','model3','model4','model5','model6','actual']]
rf2_results['predictions'] = rf2_results[['model1','model2','model3','model4','model5','model6']].mean(axis = 1)


# # output to a csv
rf2_results.to_csv('rf2_results.csv')

##### XGBoost #####
# Data frame to store results
xgb_results = pd.DataFrame({"Id":train_prices.index, "actual":train_prices.log_SalePrice})
xgb_results.set_index('Id',inplace=True)

# iterate through the 5 folds to create the first 5 models and gen results
for i in range(0,5):
    model_xgb.fit(train.iloc[train_indicies[i]], train_prices.iloc[train_indicies[i]].log_SalePrice)  # model on 80%
    pred = model_xgb.predict(train.iloc[test_indicies[i]])                              # predict on 20%
    pred = [i for i in pred]
    model_name = "model" + str(i+1)
    temp_df = pd.DataFrame({'Id':test_indicies[i], model_name:pred})    # temp DF with ID and new model results
    temp_df.set_index('Id', inplace=True)
    xgb_results = xgb_results.merge(temp_df, on='Id', how='left') # add to xgb_results DF

# create a 6th model - train on all training, predict on all test ('to_guess')
model_xgb.fit(train, train_prices.log_SalePrice)
pred = model_xgb.predict(to_guess)
pred = [i for i in pred]
temp_df = pd.DataFrame({'Id':to_guess.index, "model6":pred})
temp_df.set_index('Id', inplace=True)
xgb_results = xgb_results.merge(temp_df, on='Id', how='outer') # add to xgb_results DF

# # reorder cols (actual at the end)
#xgb_results = xgb_results[['model1','model2','model3','model4','model5','model6','actual']]
xgb_results['predictions'] = xgb_results[['model1','model2','model3','model4','model5','model6']].mean(axis = 1)


# # output to a csv
xgb_results.to_csv('xgb_results.csv')

##### GradientBoost
# Data frame to store results
gb_results = pd.DataFrame({"Id":train_prices.index, "actual":train_prices.log_SalePrice})
gb_results.set_index('Id',inplace=True)

# iterate through the 5 folds to create the first 5 models and gen results
for i in range(0,5):
    model_gb.fit(train.iloc[train_indicies[i]], train_prices.iloc[train_indicies[i]].log_SalePrice)  # model on 80%
    pred = model_gb.predict(train.iloc[test_indicies[i]])                              # predict on 20%
    pred = [i for i in pred]
    model_name = "model" + str(i+1)
    temp_df = pd.DataFrame({'Id':test_indicies[i], model_name:pred})    # temp DF with ID and new model results
    temp_df.set_index('Id', inplace=True)
    gb_results = gb_results.merge(temp_df, on='Id', how='left') # add to gb_results DF

# create a 6th model - train on all training, predict on all test ('to_guess')
model_gb.fit(train, train_prices.log_SalePrice)
pred = model_gb.predict(to_guess)
pred = [i for i in pred]
temp_df = pd.DataFrame({'Id':to_guess.index, "model6":pred})
temp_df.set_index('Id', inplace=True)
gb_results = gb_results.merge(temp_df, on='Id', how='outer') # add to gb_results DF

# # reorder cols (actual at the end)
#gb_results = gb_results[['model1','model2','model3','model4','model5','model6','actual']]
gb_results['predictions'] = gb_results[['model1','model2','model3','model4','model5','model6']].mean(axis = 1)


# # output to a csv
gb_results.to_csv('gb_results.csv')
