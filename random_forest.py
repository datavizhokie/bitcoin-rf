import math
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta, timezone
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import sys

unix_timestamp  = datetime.utcnow().timestamp()
date_format     = '%Y-%m-%d'
source_data      = "data/Bitcoin Prices with GoogleNews Avg Sentiment for '01-01-2020'-'09-10-2020'.csv"


# Spliting into training and test data
def split_data(test_perc, date_series, X, y, z):
    ''' For model training, create training and holdout data sets '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_perc, shuffle=False)
    X_train_date, X_test_date = train_test_split(date_series, test_size = test_perc, shuffle=False)
    train_price, test_price = train_test_split(z, test_size = test_perc, shuffle=False)

    train_min = X_train_date.min().strftime(date_format)
    train_max = X_train_date.max().strftime(date_format)
    pred_min = X_test_date.min().strftime(date_format)
    pred_max = X_test_date.max().strftime(date_format)

    print(f'Training set range: {train_min} - {train_max}')
    print(f'Test set range: {pred_min} - {pred_max}')

    return X_train, X_test, y_train, y_test, X_train_date, X_test_date, train_price, test_price, pred_min, pred_max


def hp_grid(n_estimator_option_num, max_depth_option_num):
    ''' Set up hyperparameter grid '''

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1600, num = n_estimator_option_num)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num = max_depth_option_num)]

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 20, 50, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 8, 10, 12]


    random_grid = { 'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

    print("Hyperparameter grid:")
    print(random_grid)

    return random_grid


def classifier_instance_iter(random_grid, n_iter, crossfolds, state, X, y, X_train, X_test, y_train, y_test):

    ''' Set up model and classifier (clf) to run iterative models based on grid
        Choose model with best performance (lowest MAE), and fit those params
    '''
    # create random forest classifier model
    rf_model = RandomForestRegressor()

    grid = ParameterGrid(random_grid)
    print(f"The total number of parameters-combinations is: {len(grid)}")

    model_cnt = n_iter*crossfolds

    print(f"Now training {n_iter} models over {crossfolds} folds of cross validation, yielding {model_cnt} models...")
    clf = RandomizedSearchCV(rf_model, random_grid, n_iter=n_iter, cv=crossfolds, random_state=state)

    # train the random search meta-estimator to find the best model out of candidates
    clf.fit(X, y)

    # print winning set of hyperparameters
    params = clf.best_estimator_.get_params()
    print('Best model parameters:')
    pprint(params)

    best_model = RandomForestRegressor(**params)
    print("Best model call:")
    print(best_model)

    # fit the best model
    best_model.fit(X_train, y_train)

    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    scores = {"Training Mean Absolute Error": mean_absolute_error(y_train, train_preds),
              "Test Mean Absolute Error": mean_absolute_error(y_test, test_preds),
              "Training R^2 score": r2_score(y_train, train_preds),
              "Test data R^2 score": r2_score(y_test, test_preds),
              "Training RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
              "Test RMSE": np.sqrt(mean_squared_error(y_test, test_preds))}

    print("Best Model Metrics:")
    print(scores)

    return train_preds, test_preds, y_train, y_test, scores


def main():
    # importing feature data
    data = pd.read_csv(source_data, skiprows=0, parse_dates=['date'])
    date_series = data['date']

    # integer-encode Signal
    conditions = [
        (data['signal'] == 'sell'),
        (data['signal'] == 'buy'),
        (data['signal'].isnull())]
    choices = [0, 1, 2]
    data['signal_int'] = np.select(conditions, choices)

    # drop lead columns that aren't necessary for features
    data = data.drop(['market_price_usd_lead_1',
                    'mining_difficulty_lead_1',
                    'hash_rate_lead_1',
                    'blockchain_txns_lead_1',
                    'unique_addresses_lead_1',
                    'buy',
                    'sell',
                    'signal'], axis=1)

    # date field -> int64 format
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].map(dt.datetime.toordinal)

    # setup X and y for modeling
    X = data.drop('diff_market_price_percent',axis=1)
    y = data['diff_market_price_percent']
    z = data['market_price_usd']
    print("Response (diff_market_price_percent) metrics:")
    print(f'Mean: {y.mean()}; Min: {y.min()}; Max: {y.max()}')

    X_train, X_test, y_train, y_test, X_train_date, X_test_date, train_price, test_price, pred_min, pred_max = split_data(0.3, date_series, X, y, z)
    
    random_grid = hp_grid(12, 8)

    #### Control scaling of training iterations here!
    train_preds, test_preds, y_train, y_test, scores = classifier_instance_iter(random_grid, 5, 3, 1, X, y, X_train, X_test, y_train, y_test) 

    # join date series with prediction results, actuals, and prices
    test_preds_df = pd.DataFrame(test_preds, columns=['y_pred'])
    y_test_df = pd.DataFrame(y_test, columns=['diff_market_price_percent']).reset_index(drop=True)
    X_test_date_df = pd.DataFrame(X_test_date).reset_index(drop=True)
    test_price_df = pd.DataFrame(test_price).reset_index(drop=True)

    test_preds_actuals= X_test_date_df.join(test_preds_df).join(y_test_df).join(test_price_df)

    # write prediction results
    file_name = f"Bitcoin %Diff Actuals, %Diff Pred, and Prices for '{pred_min}'-'{pred_max}'_unix_ts={unix_timestamp}'"
    test_preds_actuals.to_csv('data/'+file_name +'.csv', index=False)


if __name__== "__main__" :
    main()