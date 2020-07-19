# House Prices: Advanced Regression Techniques  
This is a well-known Kaggle project which contains a data set with 80 different features. These features represent different attributes of houses. Based upon these feature values, it is required to predict the final sale price of the house.  

The training data contains total 1460 training examples. A test data set is provided by Kaggle, which consists of 1459 different samples. Therefore, 1459 predictions are required to be made. 

The type of features involved and their detailed explanation can be found on [House price project](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

This project is tackled in three steps:  
- Exploratory data analysis (EDA)
- Feature engineering
- Machine learning modelling

Jupyter notebooks containing the steps and the explanation of EDA and feature engineering have been included.

## Machine Learning Modelling  
Total 4 different types of ML models are tried:  
- Ridge Regression
- Lasso Regression
- XGBRegressor
- Neural Network

In Ridge and Lasso regression, the regularization parameter 'alpha' is tuned by using 5-fold cross validation. The hyperparameters of XGBRegressor are tuned using randomized cross validation search. Moreover, a 5-layered neural network with dropout regularization is considered.  
### Findings  
For the considered problem, XGBRegressor gave the best performance. RMSE value of 0.129 was obtained by submitting the test data predictions on public leaderboard of Kaggle.
