# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
import xgboost
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# reading the preprocessed data
X_train = pd.read_csv('./data/X_train_preprocessed.csv')
y_train = pd.read_csv('./data/Y_train_preprocessed.csv')
X_test = pd.read_csv('./data/X_test_preprocessed.csv')

# reading the original test set of Kaggle to read the Id to make submission
test_orig_kaggle = pd.read_csv('./data/test.csv')
test_ids = test_orig_kaggle.Id



def rmse_evaluator(model, X_train, y_train):
    """
    Returns the root mean squared error values for each fold of cross validation.

    Parameters
    ----------
    model : sklearn machine learning model
        DESCRIPTION.
    X_train : pandas dataframe. shape = (observations, features)
    y_train : pandas dataframe. shape = (observations, 1)

    Returns
    -------
    rmse : array containing rmse values for each fold of cross validation.

    """
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, 
                                   scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

##########################################################################
#########################  Ridge Regression  #############################

print("\n############  Using Ridge Regression  #################\n")
alphas_ridge = [50, 55, 60, 62, 64, 66, 68, 70, 72, 74, 78, 85, 90]
rmse_ridge = [rmse_evaluator(Ridge(alpha = alpha), X_train, y_train).mean() 
            for alpha in alphas_ridge]

alpha_ridge_best_index = rmse_ridge.index( min(rmse_ridge) )
alpha_ridge_best = alphas_ridge[alpha_ridge_best_index]

# visualizing the alpha value vs. cross validation error.
plt.figure(figsize = (8,8))
plt.plot(alphas_ridge, rmse_ridge)
plt.xlabel('alpha value')
plt.ylabel ('cross validation error')
plt.title('Cross validation error vs. Ridge regularization')
plt.show()

# Tuned Ridge model with the best found alpha value
tuned_ridge_model = Ridge(alpha = alpha_ridge_best)
tuned_ridge_model.fit(X_train, y_train)
y_preds_ridge = tuned_ridge_model.predict(X_test)

# take exponent of the predictions because we had log transformed the y_train
# in data preprocessing
y_peds_ridge = np.squeeze(np.exp(y_preds_ridge))
submission_ridge = pd.DataFrame({'Id':test_ids, 'SalePrice':y_peds_ridge})

submission_ridge.to_csv('./data/submission_ridge.csv', index = False)


##########################################################################
#########################  Lasso Regression  #############################

print("\n############  Using Lasso Regression  #################\n")
alphas_lasso = [0.0001, 0.0005, 0.0007, 0.0008, 0.0009, 0.001, 0.0015]
rmse_lasso = [rmse_evaluator(Lasso(alpha = alpha), X_train, y_train).mean()
              for alpha in alphas_lasso]

alpha_lasso_best_index = rmse_lasso.index( min(rmse_lasso) )
alpha_lasso_best = alphas_lasso [alpha_lasso_best_index]

# visualizing the alpha value vs. cross validation error.
plt.figure(figsize = (8,8))
plt.plot(alphas_lasso, rmse_lasso)
plt.xlabel('alpha value')
plt.ylabel ('cross validation error')
plt.title('Cross validation error vs. Lasso regularization')
plt.show()

# Tuned Lasso model with the best found alpha value
tuned_lasso_model = Lasso(alpha = alpha_lasso_best)
tuned_lasso_model.fit(X_train, y_train)
y_preds_lasso = tuned_lasso_model.predict(X_test)

# take exponent of the predictions because we had log transformed the y_train
# in data preprocessing
y_peds_lasso = np.squeeze(np.exp(y_preds_lasso))
submission_lasso = pd.DataFrame({'Id':test_ids, 'SalePrice':y_peds_lasso})

submission_lasso.to_csv('./data/submission_lasso.csv', index = False)


##########################################################################
#######################  Extreme gradient boosting  ##########################

print("\n############  Using XGBRegressor  #################\n")
xgb_model = xgboost.XGBRegressor()

# Hyper Parameter Optimization
booster = ['gbtree','gblinear']
base_score = [ 0.15, 0.2, 0.23, 0.25, 0.27, 0.3]
n_estimators = [500, 800, 900, 1000, 1100]
max_depth = [1, 2, 3]
learning_rate = [0.05, 0.1, 0.15, 0.20]
min_child_weight=[1, 2, 3]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'min_child_weight':min_child_weight,
    'booster': booster,
    'base_score': base_score
    }

# Set up the grid search
random_cv = RandomizedSearchCV (estimator = xgb_model, n_iter = 200,
            param_distributions = hyperparameter_grid, cv = 5,
            scoring = 'neg_mean_absolute_error', n_jobs = 4,
            verbose = 5, random_state = 42,
            return_train_score = True)

print("\n#######  Randomized CV search for XBGRegressor hyperparameters  #########\n")
random_cv.fit(X_train,y_train)

# getting the xgb model with best parameters
best_xgboost = random_cv.best_estimator_
y_preds_xgboost = best_xgboost.predict(X_test)
# take exponent of the predictions because we had log transformed the y_train
# in data preprocessing
y_preds_xgboost = np.squeeze(np.exp(y_preds_xgboost))
submission_xgboost = pd.DataFrame({'Id':test_ids, 'SalePrice':y_preds_xgboost})
submission_xgboost.to_csv('./data/submission_xgboost.csv', index = False)


##########################################################################
#########################  Neural Network  #############################

print("\n############  Using Neural Network  #################\n")
total_features = X_train.shape[1]
nn_model = Sequential()

# layer 1
nn_model.add(Dense( 40, activation='relu', input_dim = total_features))
Dropout(0.6)

# layer 2
nn_model.add(Dense( 40, activation='relu'))
Dropout(0.6)

# layer 3
nn_model.add(Dense( 30, activation='relu'))
Dropout(0.6)

# layer 4
nn_model.add(Dense( 20, activation='relu'))
Dropout(0.6)

# layer 5 (output layer)
nn_model.add(Dense( 1))

nn_model.compile(loss = 'mean_squared_error', optimizer='adam')
model_history = nn_model.fit(X_train.values, y_train.values, epochs = 500, 
                          batch_size = 10, validation_split = 0.2, verbose = 2)

y_preds_nn = nn_model.predict(X_test.values)
# take exponent of the predictions because we had log transformed the y_train
# in data preprocessing
y_preds_nn = np.squeeze(np.exp(y_preds_nn))
submission_nn = pd.DataFrame({'Id': test_ids, 'SalePrice': y_preds_nn})
submission_nn.to_csv('./data/submission_nn.csv', index = False)