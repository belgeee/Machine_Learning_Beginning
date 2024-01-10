#general
import numpy as np
import pandas as pd
import math
#plots
import matplotlib.pyplot as plt 
import seaborn as sns
#sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error

# model = LinearRegression(fit_intercept = True)
# model


# x = np.arange(0, 8, 0.01)
# y = -1 + 3*x + np.random.normal(loc=0.0, scale=4, size=800)
# model.fit(x.reshape(-1, 1),y) # reshape for one feature design matrix


# print('Model MSE: ', metrics.mean_squared_error(y,model.predict(x.reshape(-1, 1))))



# iris = load_iris() # function to import iris data set as type "utils.Bunch" with sklearn
# X = iris.data
# y = iris.target
# feature_names = iris.feature_names
# target_names = iris.target_names
# print("Type of object iris:", type(iris))
# print("Feature names:", feature_names)
# print("Target names:", target_names)
# print("\nShape of X and y\n", X.shape, y.shape)
# print("\nType of X and y\n", type(X), type(y))



# rtree = DecisionTreeRegressor() #default setting
# print(rtree.get_params())
# # print(rtree.get_depth())