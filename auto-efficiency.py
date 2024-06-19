import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

#reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None, names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"])


#removing rows with missing values
data = data.drop(data[data['horsepower'] == '?'].index)

#separating features and target
X, y = data.iloc[:, 1:-1], data.iloc[:, 0]

#splitting the data into training and testing set using the inbuilt train_test_split function
attributes_train, attributes_test, target_train, target_test=train_test_split(X, y, random_state = 42, train_size = 0.7)

#re-indexing the dataframes to avoid index errors
attributes_train.reset_index(drop=True, inplace=True)
attributes_test.reset_index(drop=True, inplace=True)
target_train.reset_index(drop=True, inplace=True)
target_test.reset_index(drop=True, inplace=True)

#converting all numerical values to float
attributes_train["horsepower"] = attributes_train["horsepower"].astype(float)
attributes_test["horsepower"] = attributes_test["horsepower"].astype(float)
target_train = target_train.astype(float)
target_test = target_test.astype(float)
attributes_train = attributes_train.astype(float)
attributes_test = attributes_test.astype(float)


#converting the target train and test single-columned dataframes to series
target_train_ser = target_train.squeeze()
target_test_ser = target_test.squeeze()


#using the scikit-learn decision tree regressor on the data and calculating the root mean square error:
classifier_sk = tree.DecisionTreeRegressor(max_depth=4) #by default, the criterion is initialised to gini
classifier_sk = classifier_sk.fit(attributes_train, target_train)
target_pred = classifier_sk.predict(attributes_test)
rmse_sk = metrics.mean_squared_error(target_test, target_pred, squared = False)


#using our decision tree model on the same data and calculating the root mean square error:
classifier_scratch = DecisionTree(criterion="gini_index")
classifier_scratch.fit(attributes_train, target_train_ser)
target_pred_scratch = classifier_scratch.predict(attributes_test)
rmse_scratch = rmse(target_test, target_pred_scratch)


#comparing the efficiencies of the two models:
print("RMSE of scikit-learn decision tree model: ", rmse_sk)
print("RMSE of our decision tree model built from scratch: ", rmse_scratch)
