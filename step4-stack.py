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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score


def rmsle(y_pred, y_test):
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

rstacked_df = pd.read_csv('R-StackedResults.csv', index_col='Id').apply(np.log)
rstacked_df.columns = ["R Predictions"]

linear_df = pd.read_csv('linear_results.csv',
                        index_col='Id').drop(rows_to_drop, axis=1)
linear_df.columns = ["Multivariate Linear"]

gb_df = pd.read_csv('gb_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
gb_df.columns = ["Gradient Boost"]

xgb_df = pd.read_csv('xgb_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
xgb_df.columns = ["XGBoost"]

rf1_df = pd.read_csv('rf1_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
rf1_df.columns = ["Optimized Random Forest"]

rf2_df = pd.read_csv('rf2_results.csv', index_col='Id').drop(
    rows_to_drop, axis=1)
rf2_df.columns = ["Default Random Forest"]

all_predictions = linear_df.merge(gb_df, on='Id', how='inner').merge(
    xgb_df, on='Id', how='inner').merge(
    rf1_df, on='Id', how='inner').merge(
    rf2_df, on='Id', how='inner')#.merge(
    # rstacked_df, on='Id', how='inner')

actualPrices = pd.read_csv('linear_results.csv', index_col='Id')['actual']

train_predictions = all_predictions[all_predictions.index <= 1460]
actualPrices = actualPrices[actualPrices.index <= 1460]
test_predictions = all_predictions[all_predictions.index > 1460]

# Fit the predictions from different models to the actual data using GradBoost:

# Grid search for multiple hyperparameters:
boostModel = GradientBoostingRegressor()
grid_param = [{'max_depth': range(1, 4),
               'n_estimators': range(10, 1000, 50),
               'learning_rate': np.linspace(.01, .1, 10)}]
boostModel.set_params(random_state=7)
para_search = GridSearchCV(estimator=boostModel,
                           param_grid=grid_param,
                           scoring='neg_mean_squared_error',
                           cv=10, n_jobs=7,
                           return_train_score=True,
                           verbose=1)
para_search = para_search.fit(train_predictions, actualPrices)
bestModel = para_search.best_estimator_
print(bestModel)

# Fit the best model to the test data
bestModel.fit(train_predictions, actualPrices)
metapredictedTest = pd.Series(
    bestModel.predict(test_predictions)).apply(np.exp)
metapredictedTrain = pd.Series(
    bestModel.predict(train_predictions)).apply(np.exp)

############################
# Save CSV in Kaggle format
############################

outputFileName = 'stacked_predictions.csv'

# Make a list of test data IDs and Prices
id_price = list(zip(range(1461, 2920), metapredictedTest))

# Rounds price to 1 decimal place (as per sample submission example)
id_price = list(map(lambda x: [x[0], round(x[1], 1)], id_price))

# Create a dataframe
final_df = pd.DataFrame(id_price, columns=['Id', 'SalePrice'])

# Wrtie to csv
final_df.to_csv(outputFileName, index=False)

print('Saved all predictions as ' + outputFileName)

#################################################
# Print performance and graphs for various models
#################################################
actualLogPrices = actualPrices
actualPrices = np.exp(np.array(actualPrices))

clr = "#3C1E3F"
sns.set(style="darkgrid")
plt.style.use('ggplot')
plt.figure(figsize=(16, 10))
sns.set_context('talk', font_scale=1)

for model in all_predictions.columns:
    currentPredictions = train_predictions[model]
    print("For Model " + str(model))
    print("Root Mean Squared Error: $" + str(np.sqrt(mean_squared_error(
        y_pred=currentPredictions.apply(np.exp), y_true=actualLogPrices.apply(np.exp)))))
    print("Log Score: " + str(rmsle(currentPredictions.apply(np.exp),
                                    actualLogPrices.apply(np.exp))))
    grid = sns.scatterplot(x=actualLogPrices.apply(
        np.exp) / 1000, y=currentPredictions.apply(np.exp) / 1000, color=clr)
    plt.plot([0, 800], [0, 800], linewidth=2)
    plt.title(model)
    plt.xlabel("Actual price (USD/1000)")
    plt.ylabel("Predicted price (USD/1000)")
    plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.savefig('images/' + model + 'performance.png',
                bbox_inches='tight', dpi=300)
    plt.clf()

# Print results from the metamodel
print("Metamodel gives CV score of " + str(np.sqrt(-np.mean(cross_val_score(bestModel,
                                                                            train_predictions, actualPrices, cv=5, scoring='neg_mean_squared_error')))))

grid = sns.scatterplot(x=actualPrices / 1000,
                       y=metapredictedTrain / 1000, color=clr)
plt.plot([0, 800], [0, 800], linewidth=2)
plt.title('Stacked Metamodel Performance')
plt.xlabel("Actual price (USD/1000)")
plt.ylabel("Predicted price (USD/1000)")
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('images/metamodelperformance.png', bbox_inches='tight', dpi=300)
