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

# The model-building script will produce a CSV containing:
#
# | Id | model1predictions | model2predictions | model3predictions | actual
# |    |                   |                   |                   |
# |    |                   |                   |                   |
# |    |                   |                   |                   |
# |    |                   |                   |                   |

# Load training
train_df = pd.read_csv()
train_features =
train_target =

# Load test
test_features = pd.read_csv()

## Fit the predictions from different models to the actual data using GradBoost:

# Grid search for multiple hyperparameters:
boostModel = GradientBoostingRegressor()
grid_param = [{'max_depth': range(1, 4),
               'n_estimators':range(10,500,10)}]
boostModel.set_params(random_state=7)
para_search = GridSearchCV(estimator = boostModel,
                           param_grid = grid_param,
                           scoring=None,
                           cv=5, n_jobs = 7,
                           return_train_score=True,
                           verbose = 1)
para_search = para_search.fit(train_features, train_target)
bestModel = para_search.best_estimator_

# Fit the best model to the test data
bestModel.fit(train_features, train_target)
predictedTest = bestModel.predict(test_features)


results = pd.Series(predictedTest)

## Save CSV in Kaggle format
#
#test_predict is predicted prices from test set
# zips Id with price

id_price = list(zip(range(1461, 2920), np.exp(predictedTest)))

# Rounds price to 1 decimal place (as per sample submission example)
id_price = list(map(lambda x: [x[0], round(x[1], 1)], id_price))

# Create a dataframe
final_df = pd.DataFrame(id_price, columns=['Id', 'SalePrice'])

# Wrtie to csv
final_df.to_csv('submit_predictions.csv', index=False)
