#########################################
#   James' Stacking Script              #
#                                       #
#   This is a mind-bending topic, so    #
#   please excuse any confusing parts   #
#                                       #
#   The point is to use the results of  #
#   all other models to train a meta-   #
#   model, which will be used to make   #
#   final predictions on the test set   #
#                                       #
#########################################

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error

def rmsle(y_pred, y_test) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(y_pred) - np.log(y_test))**2))

# The model-building script will produce a CSV containing:
#
# | Id | model1predictions | model2predictions | model3predictions | actual |
# |    |                   |                   |                   |        |
# |    |                   |                   |                   |        |
# |    |                   |                   |                   |        |
# |    |                   |                   |                   |        |

# Load training and test predictions
rows_to_drop = ['model1', 'model2', 'model3',
                'model4', 'model5', 'model6', 'actual']

linear_df = pd.read_csv('linear_results.csv',
                        index_col='Id').drop(rows_to_drop, axis=1)
linear_df.columns = ["LinearPredictions"]

gb_df = pd.read_csv('gb_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
gb_df.columns = ["GBoostPredictions"]

xgb_df = pd.read_csv('xgb_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
xgb_df.columns = ["XGBoostPredictions"]

rf1_df = pd.read_csv('rf1_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
rf1_df.columns = ["RandForest1Predictions"]

rf2_df = pd.read_csv('rf2_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
rf2_df.columns = ["RandForest2Predictions"]

all_predictions = linear_df.merge(gb_df, on='Id', how='inner').merge(
    xgb_df, on='Id', how='inner').merge(rf1_df, on='Id', how='inner').merge(rf2_df, on='Id', how='inner')

actualPrices = pd.read_csv('linear_results.csv', index_col='Id')['actual']

train_predictions = all_predictions[all_predictions.index <= 1460]
actualPrices = actualPrices[actualPrices.index <= 1460]
test_predictions = all_predictions[all_predictions.index > 1460]

# Fit the predictions from different models to the actual data using GradBoost:

# Grid search for multiple hyperparameters:
boostModel = GradientBoostingRegressor()
grid_param = [{'max_depth': range(1, 4),
               'n_estimators': range(10, 500, 100),
               'learning_rate': np.linspace(.01,.1,2)}]
boostModel.set_params(random_state=7)
para_search = GridSearchCV(estimator=boostModel,
                           param_grid=grid_param,
                           scoring=None,
                           cv=5, n_jobs=7,
                           return_train_score=True,
                           verbose=1)
para_search = para_search.fit(train_predictions, actualPrices)
bestModel = para_search.best_estimator_

# Fit the best model to the test data
bestModel.fit(train_predictions, actualPrices)
predictedTest = pd.Series(bestModel.predict(test_predictions)).apply(np.exp)
predictedTrain = pd.Series(bestModel.predict(train_predictions)).apply(np.exp)

############################
# Save CSV in Kaggle format
############################

outputFileName = 'stacked_predictions.csv'

# Make a list of test data IDs and Prices
id_price = list(zip(range(1461, 2920), predictedTest))

# Rounds price to 1 decimal place (as per sample submission example)
id_price = list(map(lambda x: [x[0], round(x[1], 1)], id_price))

# Create a dataframe
final_df = pd.DataFrame(id_price, columns=['Id', 'SalePrice'])

# Wrtie to csv
final_df.to_csv(outputFileName, index=False)

print('Saved all predictions as ' + outputFileName)

####################
# Print performance
####################

for model in all_predictions.columns:
    currentPredictions = all_predictions[model][:1457]
    print("For Model " + str(model))
    print("Root Mean Squared Error: $" + str(np.sqrt(mean_squared_error(y_pred=currentPredictions.apply(np.exp), y_true=actualPrices.apply(np.exp)))))
    print("Log Score: " + str(rmsle(currentPredictions.apply(np.exp), actualPrices.apply(np.exp))))

print('For metamodel:')
print("Root Mean Squared Error: $" + str(np.sqrt(mean_squared_error(y_pred=predictedTrain, y_true=actualPrices.apply(np.exp)))))
print("Log Score: " + str(rmsle(y_pred = predictedTrain, y_test = actualPrices.apply(np.exp))))
