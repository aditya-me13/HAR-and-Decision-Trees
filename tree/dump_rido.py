import numpy as np
import pandas as pd

class DecisionTreeSplit:
    def __init__(self, feature_index, split_value):
        self.feature_index = feature_index
        self.split_value = split_value

def df_to_array(attributes):
  X = attributes.to_numpy()
  return X

def entropy_(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_value

def find_optimal_split(X, y):
    sorted_indices = np.argsort(X)
    sorted_X = X[sorted_indices]
    sorted_y = y[sorted_indices]

    change_indices = np.where(np.diff(sorted_y) != 0)[0] + 1

    if len(change_indices) == 0:
        return None

    optimal_split = None
    min_entropy = float('inf')

    for idx in change_indices:

        split_value = np.mean(sorted_X[idx - 1:idx + 1])
        left_mask = X <= split_value
        right_mask = ~left_mask


        left_entropy = entropy_(y[left_mask])
        right_entropy = entropy_(y[right_mask])
        weighted_entropy = (len(y[left_mask]) * left_entropy + len(y[right_mask]) * right_entropy) / len(y)

        if weighted_entropy < min_entropy:
            min_entropy = weighted_entropy
            optimal_split = DecisionTreeSplit(None, split_value)
    return optimal_split

def find_optimal_attribute(X, y):
    num_features = X.shape[1]
    optimal_split = None
    min_entropy = float('inf')

    for feature_index in range(num_features):
        current_feature = X[:, feature_index]
        current_optimal_split = find_optimal_split(current_feature, y)

        if current_optimal_split is not None and current_optimal_split.split_value is not None:
            current_entropy = entropy_(y[current_feature <= current_optimal_split.split_value])
            current_entropy += entropy_(y[current_feature > current_optimal_split.split_value])

            if current_entropy < min_entropy:
                min_entropy = current_entropy
                optimal_split = DecisionTreeSplit(feature_index, current_optimal_split.split_value)

    return optimal_split


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# np.random.seed(42)


# X, y = make_classification(
#     n_features=5, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# optimal_attribute = find_optimal_attribute(df_to_array(attributes, output))
# # print(X, y)

# if optimal_attribute is not None:
#     print("Optimal Split Feature Index:",  df.columns[optimal_attribute.feature_index]optimal_attribute.feature_index)
#     print("Optimal Split Value:", optimal_attribute.split_value)
# else:
#     print("No optimal split found.")

#metrics.py
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


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size > 0
    assert y.size > 0
    assert y_hat.size == y.size
    return sum([actual == cls and pred == cls for actual, pred in zip(y, y_hat)])/sum([actual == cls for actual in y])    


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size > 0
    assert y.size > 0
    assert y_hat.size == y.size
    return (sum([(pred - actual)**2 for pred, actual in zip(y_hat, y)]) / y.size)**0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size > 0
    assert y.size > 0
    assert y_hat.size == y.size
    return sum([abs(pred - actual) for pred, actual in zip(y_hat, y)]) / y.size



N = 30
P = 5
np.random.seed(42)
random_values = np.random.choice([0, 1], size=N)
y = pd.Series(random_values)
X = pd.DataFrame(np.random.randn(N, P))
#print(X,y)

optimal_attribute = find_optimal_attribute(df_to_array(X), np.array(y))
# if optimal_attribute is not None:
#     print("Optimal Split Feature Index:",  X.columns[optimal_attribute.feature_index])
#     print("Optimal Split Value:", optimal_attribute.split_value)
# else:
#     print("No optimal split found.")
print(optimal_attribute.split_value)

import pandas as pd

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    assert y.size > 0
    if y.dtype == "category":
        return False
    return True

# Splitting the data into left and right
def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    assert y.size > 0
    assert len(X) == y.size

    # Discrete Input
    if(not check_ifreal(X[attribute])):
        to_return = []
        unique_vals = X[attribute].unique()

        for unique in unique_vals:

            mask = (X[attribute]==unique)
            X_mask = X[mask]
            y_mask = y[mask]
            to_return.append((X_mask, y_mask))

        return to_return

    # Real Input
    # Pass the value to be spliited
    # Only binary split possible
    else:

        mask1= (X[attribute] <= value)
        mask2= (X[attribute] > value)
        X_mask = X[mask1]
        y_mask = y[mask1]

        Xn_mask = X[mask2]
        yn_mask = y[mask2]

        to_return = (X_mask, y_mask), (Xn_mask, yn_mask)

        return to_return
    

# Building the tree

class Node:
    def __init__(self, to_split=None, value=None, depth=float('inf'), decision=None, split_val=0, left=None,
                 right=None):
        self.to_split = to_split
        self.split_val = split_val
        self.depth = depth
        self.decision = decision
        self.left_ptr = None
        self.right_ptr = None


class DecisionTree:
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion='information_gain', max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.Root = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        def dfconvertor(a):
            x = pd.DataFrame(a[0])
            x[a[1].name] = a[1]
            x = x.reset_index(drop=True)
            return x.iloc[:, :-1], x.iloc[:, -1]

        def make_tree(X: pd.DataFrame, y: pd.Series, features: list, criterion: str, max_depth: int, node_):

            if y.nunique() == 1:
                node_.decision = y.mode()[0]
                return
            if len(features) == 0:
                node_.decision = y.mode()[0]
                return
            elif max_depth == 0:
                node_.decision = y.mode()[0]
                return
            else:
                optimal_attribute = find_optimal_attribute(df_to_array(X), np.array(y))
                opt_attribute = X.columns[optimal_attribute.feature_index]
                split_val = optimal_attribute.split_value
                # print(opt_attribute, split_val)
                node_.to_split = opt_attribute
                node_.split_val = split_val
                t, r = split_data(X, y, opt_attribute, split_val)
                left_df_X, left_df_y = dfconvertor(t)
                right_df_X, right_df_y = dfconvertor(r)

                new_node = Node(depth=node_.depth + 1)
                node_.left_ptr = new_node
 
                features_ = [feature for feature in features if feature != opt_attribute]
                make_tree(left_df_X, left_df_y, features_, criterion, max_depth - 1, new_node)

                new_node_ = Node(depth=node_.depth + 1)
                node_.right_ptr = new_node_

                _features_ = [feature for feature in features if feature != opt_attribute]
                make_tree(right_df_X, right_df_y, _features_, criterion, max_depth - 1, new_node_)

            return

        self.Root = Node(depth=0)
        make_tree(X, y, list(X.columns), self.criterion, self.max_depth - 1, self.Root)
        return None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """

        def predict_from_node(dataFrame, root):
            ## Do check out for len(dataFrame.columns) == 0 condition
            if (root.decision is not None):
                return root.decision

            split_attribute = root.to_split
            split_value = root.split_val

            if dataFrame[split_attribute][0] <= split_value:
                return predict_from_node(dataFrame, root.left_ptr)

            else:
                return predict_from_node(dataFrame, root.right_ptr)

        def predict_(X: pd.DataFrame, tree_root):
            y = []
            for i in range(X.shape[0]):
                df = pd.DataFrame(X.iloc[i, :]).T
                df.reset_index(inplace=True, drop=True)
                y.append(predict_from_node(df, tree_root))
            return pd.Series(y)

        return predict_(X, self.Root)
    
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
            if (node_.decision is not None):
                print("   "*depth,node_.decision)
                return None
            else: # node_decision is None
                print("   "*depth,'?(', node_.to_split, '<=', node_.split_val, ')', sep='')
                plot_(node_.left_ptr, depth+1)

                print("   "*depth,'?(', node_.to_split, '>', node_.split_val, ')', sep='')
                plot_(node_.right_ptr, depth+1)
        
        plot_(self.Root, 0)
        return None    

# N = 30
# P = 5
# np.random.seed(42)
# random_values = np.random.choice([0, 1], size=N)
# y = pd.Series(random_values)
# X = pd.DataFrame(np.random.randn(N, P))

# DT = DecisionTree()
# DT.fit(X, y)
# predictions = DT.predict(X)
# print(predictions)
    
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")
tree = DecisionTree()  # Split based on Inf. Gain
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()

# for criteria in ["information_gain"]:
#     tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
#     tree.fit(X, y)
#     y_hat = tree.predict(X)
#     # tree.plot()
#     print("Criteria :", criteria)
#     print("Accuracy: ", accuracy(y_hat, y))
#     for cls in y.unique():
#         print("Precision: ", precision(y_hat, y, cls))
#         print("Recall: ", recall(y_hat, y, cls))    


# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics

# DT = DecisionTreeClassifier()
# DT.fit(X,y)
# y_hat = DT.predict(X)
# print("accuracy", metrics.accuracy_score(y, y_hat))
# for cls in y.unique():
#         print("Precision: ", metrics.precision(y_hat, y, cls))
#         print("Recall: ", metrics.recall(y_hat, y, cls)) 