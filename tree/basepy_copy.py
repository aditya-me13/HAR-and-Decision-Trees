"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_final import *

np.random.seed(42)


@dataclass

class Next:
    def __init__(self,name, ptr = None):
        self.ptr = ptr
        self.name = name
class Node:
    def __init__(self, to_split=None, value=0, depth=float('inf'), decision = None):
        self.to_split = to_split
        self.value = value
        self.depth = depth
        self.decision = decision
        self.nexts = []

    def add_in_nexts(self,name, ptr=None):
        self.nexts.append(Next(name = name))
        # lst = self.nexts
        # lst.append(Next(name = name,ptr = ptr))
        # self.nexts = lst

class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion = "information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.Root = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        def dfconvertor(a):
            x=pd.DataFrame(a[0])
            x[a[1].name]=a[1]
            x=x.reset_index(drop=True)
            return x.iloc[:,:-1], x.iloc[:,-1]
        
        def make_tree(X: pd.DataFrame, y:pd.Series, features:list, criterion:str, max_depth:int, node_):
            if(y.nunique() == 1):
                node_.decision = y.mode()[0]
                return
            elif(len(features) == 0 and (not check_ifreal(y))):
                node_.decision = y.mode()[0]
                return
            elif(len(features) == 0 and (check_ifreal(y))):
                node_.decision = y.mean()#[0]
                return
            elif (node_.depth > max_depth and (not check_ifreal(y))):
                node_.decision = y.mode()[0]
                return
            elif (node_.depth > max_depth and (check_ifreal(y))):
                node_.decision = y.mean()
                return
            else:
                opt_attr_ = opt_split_attribute(X, y, criterion, pd.Series(features))
                node_.to_split = opt_attr_
                for element in split_data(X, y, opt_attr_, 0):
                    Xi, yi = dfconvertor(element) 
                    node_.add_in_nexts(name = Xi[opt_attr_][0])
                    new_node = Node(depth = node_.depth + 1)
                    node_.nexts[-1].ptr = new_node 
                    features_ = [feature for feature in features if feature != opt_attr_]
                    make_tree(Xi, yi, features_, criterion, max_depth, new_node)
                return 
            
        self.Root = Node(depth = 0)   
        make_tree(X, y, list(X.columns), self.criterion, self.max_depth-1, self.Root)
        return None


        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        def predict_from_node(dataFrame, root):
            ## Do check out for len(dataFrame.columns) == 0 condition
            if (root.decision is not None):
                return root.decision

            split_attribute = root.to_split
            list_ = dataFrame[split_attribute].unique()

            for _ in list_:
                for unique in root.nexts:
                    if (_ == unique.name):
                        return predict_from_node(dataFrame, unique.ptr)
  

        def predict_(X: pd.DataFrame, tree_root):
            y = []
            for i in range(X.shape[0]):
                df = pd.DataFrame(X.iloc[i, :]).T
                df.reset_index(inplace=True, drop=True)
                y.append(predict_from_node(df, tree_root))
            return pd.Series(y)            

        return predict_(X, self.Root)        

        # def predict_from_node(feature_names: list, feature_values: list, node):
        #     if (node.decision is not None) or (len(feature_names)==0):
        #         return node.decision
        #     else:
        #         split_attribute = node.to_split
        #         val = feature_values[feature_names.index(split_attribute)]
        #         unique_list = node.nexts #.copy()
        #         for unique in unique_list:
        #             if (unique.name ==  val):
        #                 feature_values_ = [element for i, element in enumerate(feature_values) if i != feature_names.index(split_attribute)]
        #                 feature_names_ = [element for i, element in enumerate(feature_names) if element != split_attribute]
        #                 return predict_from_node(feature_names_, feature_values_, unique.ptr)
        #         return None


        # def predict_(X: pd.DataFrame, Tree_Root):
        #     feature_names = X.columns
        #     y = []
        #     for iter, row in X.iterrows():
        #         y.append(predict_from_node(list(feature_names), list(row), Tree_Root))
        #     return pd.Series(y)
        # # Traverse the tree you constructed to return the predicted values for the given test inputs.

        # return predict_(X, self.Root)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        def plot_(node_, depth):
            if (node_.to_split is None):
                print(node_.decision)
                return None
            else: # node_decision is None
                print('?(',node_.to_split,')', sep='')
                for element in node_.nexts:
                    print("   "*depth, element.name, ': ',end = '', sep = '')
                    plot_(element.ptr, depth+1)
        
        plot_(self.Root, 1)
        return None
    

# df = pd.read_csv("Tennis.csv")
# X_train = df.iloc[:11,:-1]
# y_train = df.iloc[:11,-1]
# X_test = df.iloc[11:,:-1]
# y_test = df.iloc[11:,-1]

# for criteria in ["information_gain", 'gini_index']:
#     DT = DecisionTree(criterion = criteria)  # Split based on Inf. Gain
#     DT.fit(X_train, y_train)
#     y_hat = DT.predict(X_test)
#     print(y_hat)
    # print("Criteria :", criteria)
    # print("Accuracy: ", accuracy(y_hat, y_test))
    # for cls in y_test.unique():
    #     print("Precision: ", precision(y_hat, y_test, cls))
    #     print("Recall: ", recall(y_hat, y_test, cls))    

# def rmse(y_hat: pd.Series, y: pd.Series) -> float:
#     """
#     Function to calculate the root-mean-squared-error(rmse)
#     """
#     assert y_hat.size > 0
#     assert y.size > 0
#     assert y_hat.size == y.size
#     return (sum([(pred - actual)**2 for pred, actual in zip(y_hat, y)]) / y.size)**0.5


# def mae(y_hat: pd.Series, y: pd.Series) -> float:
#     """
#     Function to calculate the mean-absolute-error(mae)
#     """
#     assert y_hat.size > 0
#     assert y.size > 0
#     assert y_hat.size == y.size
#     return sum([abs(pred - actual) for pred, actual in zip(y_hat, y)]) / y.size


# N = 30
# P = 5
# X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randn(N))

# tree = DecisionTree()
# tree.fit(X.iloc[:21,:], y[:21])
# y_hat = tree.predict(X.iloc[21:,:])
# print(y_hat)
# print(y[21:])
# print("RMSE: ", rmse(y_hat, y[21:]))
# print("MAE: ", mae(y_hat, y[21:]))

# for criteria in ["information_gain", "gini_index"]:
#     tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
#     tree.fit(X, y)
#     y_hat = tree.predict(X)
#     # tree.plot()
#     print("Criteria :", criteria)
#     print("RMSE: ", rmse(y_hat, y))
#     print("MAE: ", mae(y_hat, y))
    
from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    assert y_hat.size > 0
    assert y.size > 0
    return sum([actual == pred for actual, pred in zip(y, y_hat)]) / y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float|None:
    """
    Function to calculate the precision
    """
    assert y_hat.size > 0
    assert y.size > 0
    assert y_hat.size == y.size
    if sum([pred == cls for pred in y_hat]) != 0:
        return sum([actual == cls and pred == cls for actual, pred in zip(y, y_hat)])/sum([pred == cls for pred in y_hat])
    else:
        print(f'No predictions were made for class {cls}. Hence, precision for class {cls} is not defined.')
        return None

N = 30
P = 5
# X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randint(P, size=N), dtype="category")

X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X.iloc[:21,:], y[:21])
    y_hat = tree.predict(X.iloc[21:,:])
    print(y_hat)
    print(y[21:])
    # tree.plot()
    # print("Criteria :", criteria)
    # print("Accuracy: ", accuracy(y_hat, y[21:]))
    # for cls in y.unique():
    #     print("Precision: ", precision(y_hat, y[21:], cls))
    #     print("Recall: ", recall(y_hat, y, cls))