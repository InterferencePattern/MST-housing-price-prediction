from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import time

def find_hyperparams(model, train_path, target_path, n_jobs=1):
    """This function tunes hyperparameters for one of three types of models (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor).
    The resulting dictionary containing 'best' hyperparameters is saved as a pickle file.

    Keyword Arguments
    -----------------
    model -- The model to tune hyperparameters for. Must be one of ['rf', 'gb', 'xgb']
    train_path -- The file path for the training set (train set should not contain the target)
    target_path -- The file path for the target
    n_jobs = Number of cpu cores to use for computation, use -1 for all
    """

    # Load train and target sets
    train = pd.read_csv(train_path, index_col='Id')
    target = pd.read_csv(target_path, index_col='Id')

    # Check if model argument is one of the correct model types
    if model in ['rf', 'gb', 'xgb']:
        # If model = rf, run RandomForest tuning and write resulting dict to a pickle
        if model == 'rf':
            best_hyperparams = rf_tune(train, target, n_jobs)
            write_pkl(best_hyperparams, 'rf_params.pkl')
            print('Your RandomForest model hyperparamters have been saved as "rf_params.pkl".')
        # If model = gb, run GradientBoosting tuning and write resulting dict to a pickle
        elif model == 'gb':
            best_hyperparams = gb_tune(train, target, n_jobs)
            write_pkl(best_hyperparams, 'gb_params.pkl')
            print('Your GradientBoosting model hyperparamters have been saved as "gb_params.pkl".')
        # If model = xgb, run XGBoost tuning and write resulting dict to a pickle
        else:
            best_hyperparams = xgb_tune(train, target, n_jobs)
            write_pkl(best_hyperparams, 'xgb_params.pkl')
            print('Your XGBoost model hyperparamters have been saved as "xgb_params.pkl".')

    # If model argument not correct form, alert the user
    else:
        print('{} is not a valid input for the model argument. Please enter "rf" for RandomForest, "gb" for GradientBoosting, or "xgb" for XGBoost.'.format(model))

def write_pkl(my_dict, opt_path):
    """Writes a dictionary to a pickle

    Keyword arguments
    -----------------
    my_dict -- A dictionary
    opt_path -- Name to save the pickle as
    """
    # Save dictionary to pickle
    with open(opt_path, 'wb') as f:
        pickle.dump(my_dict, f)

def rf_tune(train, target, n_jobs):
    """Tunes hyperparameters for a RandomForestRegressor model.

    Keyword Arguments
    -----------------
    train -- Training set as pandas DataFrame (without target)
    target -- Target as pandas DataFrame
    n_jobs -- Number of cpu cores to use for computation, use -1 for all
    """

    from sklearn.ensemble import RandomForestRegressor
    #
    model = RandomForestRegressor(bootstrap=False, max_features='sqrt')
    #
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    # Maximum number of nodes in each tree
    max_depth = [int(x) for x in np.linspace(10, 50, num = 9)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3]

    # Store in a dictionary
    random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}

    # Make a RandomizedSearchCV object with correct model and specified hyperparams
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2, random_state=42, n_jobs=n_jobs)
    start = time.time()
    # Fit models
    rf_random.fit(train, target.values.ravel())
    print('It took {} minutes to run the RandomizedSearchCV'.format(round((int(time.time() - start)/60)), 2))

    # Add bootstrap = False to the list of best hyperparameters
    best_params = rf_random.best_params_
    best_params['bootstrap'] = False
    best_params['max_features'] = 'sqrt'

    print('Here were the best hyperparameters:\n\n{}'.format(rf_random.best_params_))

    return best_params


def gb_tune(train, target, n_jobs):
    """Tunes hyperparameters for a GradientBoostingRegressor model.

    Keyword Arguments
    -----------------
    train -- Training set as pandas DataFrame (without target)
    target -- Target as pandas DataFrame
    n_jobs -- Number of cpu cores to use for computation, use -1 for all
    """
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(max_features='sqrt')

    # Amount to weight the contrbution of each additional tree
    learning_rate = [round(float(x), 2) for x in np.linspace(start = .01, stop = .1, num = 10)]
    # Number of boosting stages to perform
    n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 5000, num = 10)]
    # Maximum nodes in each tree
    max_depth = [1, 2, 3, 4, 5]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 8, 10, 12]
    # Minimum number of samples required for a node
    min_samples_leaf = [1, 2, 3, 4, 5]

    random_grid = {'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}

    # Make a RandomizedSearchCV object with correct model and specified hyperparams
    gb_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2, random_state=42, n_jobs=n_jobs)
    start = time.time()
    # Fit models
    gb_random.fit(train, target.values.ravel())
    print('It took {} minutes to run the RandomizedSearchCV'.format(round((int(time.time() - start)/60)), 2))
    print('Here were the best hyperparameters:\n\n{}'.format(gb_random.best_params_))

    best_params = gb_random.best_params_
    best_params['max_features'] = 'sqrt'

    return best_params

def xgb_tune(train, target, n_jobs):
    """Tunes hyperparameters for a XGBoostRegressor model.

    Keyword Arguments
    -----------------
    train -- Training set as pandas DataFrame (without target)
    target -- Target as pandas DataFrame
    n_jobs -- Number of cpu cores to use for computation, use -1 for all
    """
    import xgboost as xgb

    model = xgb.XGBRegressor()

    # Amount to weight the contrbution of each additional tree
    learning_rate = [round(float(x), 2) for x in np.linspace(start = .1, stop = .2, num = 11)]
    # Minimum for sum of weights for observations in a node
    min_child_weight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Maximum nodes in each tree
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]

    random_grid = {'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight}

    # Make a RandomizedSearchCV object with correct model and specified hyperparams
    xgb_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=1000, cv=5, verbose=2, random_state=42, n_jobs=n_jobs)
    start = time.time()
    # Fit models
    xgb_random.fit(train, target)
    print('It took {} minutes to run the RandomizedSearchCV'.format(round((int(time.time() - start)/60)), 2))
    print('Here were the best hyperparameters:\n\n{}'.format(xgb_random.best_params_))

    return xgb_random.best_params_
=======
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import time

def find_hyperparams(model, train_path, target_path, n_jobs=1):
    """This function tunes hyperparameters for one of three types of models (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor).
    The resulting dictionary containing 'best' hyperparameters is saved as a pickle file.

    Keyword Arguments
    -----------------
    model -- The model to tune hyperparameters for. Must be one of ['rf', 'gb', 'xgb']
    train_path -- The file path for the training set (train set should not contain the target)
    target_path -- The file path for the target
    n_jobs = Number of cpu cores to use for computation, use -1 for all
    """

    # Load train and target sets
    train = pd.read_csv(train_path, index_col='Id')
    target = pd.read_csv(target_path, index_col='Id')

    # Check if model argument is one of the correct model types
    if model in ['rf', 'gb', 'xgb']:
        # If model = rf, run RandomForest tuning and write resulting dict to a pickle
        if model == 'rf':
            best_hyperparams = rf_tune(train, target, n_jobs)
            write_pkl(best_hyperparams, 'rf_params.pkl')
            print('Your RandomForest model hyperparamters have been saved as "rf_params.pkl".')
        # If model = gb, run GradientBoosting tuning and write resulting dict to a pickle
        elif model == 'gb':
            best_hyperparams = gb_tune(train, target, n_jobs)
            write_pkl(best_hyperparams, 'gb_params.pkl')
            print('Your GradientBoosting model hyperparamters have been saved as "gb_params.pkl".')
        # If model = xgb, run XGBoost tuning and write resulting dict to a pickle
        else:
            best_hyperparams = xgb_tune(train, target, n_jobs)
            write_pkl(best_hyperparams, 'xgb_params.pkl')
            print('Your XGBoost model hyperparamters have been saved as "xgb_params.pkl".')

    # If model argument not correct form, alert the user
    else:
        print('{} is not a valid input for the model argument. Please enter "rf" for RandomForest, "gb" for GradientBoosting, or "xgb" for XGBoost.'.format(model))

def write_pkl(my_dict, opt_path):
    """Writes a dictionary to a pickle

    Keyword arguments
    -----------------
    my_dict -- A dictionary
    opt_path -- Name to save the pickle as
    """
    # Save dictionary to pickle
    with open(opt_path, 'wb') as f:
        pickle.dump(my_dict, f)

def rf_tune(train, target, n_jobs):
    """Tunes hyperparameters for a RandomForestRegressor model.

    Keyword Arguments
    -----------------
    train -- Training set as pandas DataFrame (without target)
    target -- Target as pandas DataFrame
    n_jobs -- Number of cpu cores to use for computation, use -1 for all
    """

    from sklearn.ensemble import RandomForestRegressor
    #
    model = RandomForestRegressor()
    #
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of nodes in each tree
    max_depth = [int(x) for x in np.linspace(10, 50, num = 9)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 8, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4, 5]

    # Store in a dictionary
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}

    # Make a RandomizedSearchCV object with correct model and specified hyperparams
    rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=n_jobs)
    start = time.time()
    # Fit models
    rf_random.fit(train, target.values.ravel())
    print('It took {} minutes to run the RandomizedSearchCV'.format(round((int(time.time() - start)/60)), 2))

    # Add bootstrap = False to the list of best hyperparameters
    best_params = rf_random.best_params
    best_params['bootstrap'] = False

    print('Here were the best hyperparameters:\n\n{}'.format(rf_random.best_params_))

    return best_params


def gb_tune(train, target, n_jobs):
    """Tunes hyperparameters for a GradientBoostingRegressor model.

    Keyword Arguments
    -----------------
    train -- Training set as pandas DataFrame (without target)
    target -- Target as pandas DataFrame
    n_jobs -- Number of cpu cores to use for computation, use -1 for all
    """
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor()

    # Amount to weight the contrbution of each additional tree
    learning_rate = [round(float(x), 2) for x in np.linspace(start = .01, stop = .2, num = 10)]
    # Number of boosting stages to perform
    n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 5000, num = 10)]
    # Maximum nodes in each tree
    max_depth = [1, 2, 3, 4, 5]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6, 8, 10]
    # Minimum number of samples required for a node
    min_samples_leaf = [1, 2, 3, 4, 5]
    # Number of features to consider when looking for best node split
    max_features = ['auto', 'sqrt']

    random_grid = {'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features}

    # Make a RandomizedSearchCV object with correct model and specified hyperparams
    gb_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=n_jobs)
    start = time.time()
    # Fit models
    gb_random.fit(train, target.values.ravel())
    print('It took {} minutes to run the RandomizedSearchCV'.format(round((int(time.time() - start)/60)), 2))
    print('Here were the best hyperparameters:\n\n{}'.format(gb_random.best_params_))

    return gb_random.best_params_

def xgb_tune(train, target, n_jobs):
    """Tunes hyperparameters for a XGBoostRegressor model.

    Keyword Arguments
    -----------------
    train -- Training set as pandas DataFrame (without target)
    target -- Target as pandas DataFrame
    n_jobs -- Number of cpu cores to use for computation, use -1 for all
    """
    import xgboost as xgb

    model = xgb.XGBRegressor()

    # Amount to weight the contrbution of each additional tree
    learning_rate = [round(float(x), 2) for x in np.linspace(start = .05, stop = .2, num = 10)]
    # Minimum for sum of weights for observations in a node
    min_child_weight = [1, 2, 3, 4, 5]
    # Maximum nodes in each tree
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]

    random_grid = {'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight}

    # Make a RandomizedSearchCV object with correct model and specified hyperparams
    xgb_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=n_jobs)
    start = time.time()
    # Fit models
    xgb_random.fit(train, target)
    print('It took {} minutes to run the RandomizedSearchCV'.format(round((int(time.time() - start)/60)), 2))
    print('Here were the best hyperparameters:\n\n{}'.format(xgb_random.best_params_))

    return xgb_random.best_params_
>>>>>>> upstream/master
