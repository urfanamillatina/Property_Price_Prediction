# Property Price Prediction

Modelling Machine Learning for Property Sales Price Prediction with deployment in here: https://property-price-prediction-335eaff00c71.herokuapp.com/

![My Website Demo](./assets/website-demo.gif)

## What's this model about?

This machine learning model predicts house sales price  which falls into regression model, supervised learning category. Supervised learning aims to detect the relationships between independent features that belong to a property and a target feature which in this case, is the Sales price of the property. 

The datasets used in this model: data.csv, train.csv and test.csv which are acquired from https://www.kaggle.com/competitions/home-data-for-ml-course/data.


## The Software and Tools
1. [GithubAccount](https://github.com) for repository
2. [HerokuAccount](https://heroku.com) for web deployment
3. [VSCodeIDE](https://code.visualstudio.com/) with Jupyter Notebook extension for development
4. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line) for git commands
5. [Postman](https://www.postman.com/) for the API

## EDA:
From 80 independent features, only 18 features have strong correlations ≥0.5 or ≤-0.5 with the target feature 'SalePrice'. Thus, the rest of the features are dropped.

# Data Pre-Processing:
1. One-hot-encoding to convert categorical feature to numerical feature
2. Using StandarScaler to scale the values from all variables to 0.

## Modelling
This machine learning applies three algorithms: 
1. Support Vector Regression (SVR)
2. Random Forest
3. eXtreme Gradient Boost (XGBoost).

## Evaluation
Performance metrics:
1. RMSE
2. R-Squared
3. MAE
4. MSE

With these performance metrics, the algorithm Random Forest is the one with higest scores across the board.

## Deployment
Deployment process:
1. Using Postman to POST API
2. Commit and push all files and changes into Github Repository
3. Web deployment through heroku: https://property-price-prediction-335eaff00c71.herokuapp.com/


