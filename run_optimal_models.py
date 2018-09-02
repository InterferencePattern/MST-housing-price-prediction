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
from sklearn.model_selection import cross_val_score

def run_a_model(modelObject):
    # Data frame to store results
    current_results = pd.DataFrame({"Id":train_prices.index, "actual":train_prices.log_SalePrice})
    current_results.set_index('Id',inplace=True)
    # iterate through the 5 folds to create the first 5 models and gen results
    for i in range(0,5):
        modelObject.fit(train.iloc[train_indicies[i],], train_prices.iloc[train_indicies[i],].values)  # model on 80%
        pred = modelObject.predict(train.iloc[test_indicies[i]])                              # predict on 20%
        pred = [i for i in pred]
        model_name = "model" + str(i+1)
        temp_df = pd.DataFrame({'Id':train.index[test_indicies[i]], model_name:pred})    # temp DF with ID and new model results
        temp_df.set_index('Id', inplace=True)
        current_results = current_results.merge(temp_df, on='Id', how='left', left_index = True) # add to current results DF
    # create a 6th model - train on all training, predict on all test ('to_guess')
    modelObject.fit(train, train_prices.values)
    pred = modelObject.predict(to_guess)
    pred = [i for i in pred]
    temp_df = pd.DataFrame({'Id':to_guess.index, "model6":pred})
    temp_df.set_index('Id', inplace=True)
    current_results = current_results.merge(temp_df, on='Id', how='outer') # add to current_results DF
    score = str(np.sqrt(-np.mean(cross_val_score(modelObject, train, train_prices, cv=5, scoring = 'neg_mean_squared_error'))))
    current_results['predictions'] = current_results[['model1','model2','model3','model4','model5','model6']].mean(axis = 1)
    return current_results, score

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
kf = KFold(n_splits=5, shuffle = True, random_state = 7)

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
linear_results, score = run_a_model(model_linear)
print("Linear model gives CV score of " + str(score))
# output to a csv
linear_results.to_csv('linear_results.csv')

###### Random Forest 1 #####
rf1_results, score = run_a_model(model_rf1)
print("Random forest 1 gives CV score of " + str(score))
# # output to a csv
rf1_results.to_csv('rf1_results.csv')

##### Random Forest 2 #####
rf2_results, score = run_a_model(model_rf2)
print("Random forest 2 gives CV score of " + str(score))
# # output to a csv
rf2_results.to_csv('rf2_results.csv')

##### XGBoost #####
# Data frame to store results
xgb_results, score = run_a_model(model_xgb)
print("XGBoost gives CV score of " + str(score))
# # output to a csv
xgb_results.to_csv('xgb_results.csv')

##### GradientBoost
gb_results, score = run_a_model(model_gb)
print("Gradient Boost gives CV score of " + str(score))
# # output to a csv
gb_results.to_csv('gb_results.csv')
