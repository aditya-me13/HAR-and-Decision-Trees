"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from scipy.special import xlogy
import statistics

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    assert y.size > 0
    if y.dtype == "category" or y.dtype == "object":
        return False
    return True


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    assert Y.size > 0

    probabilities = Y.value_counts() / Y.size
    
    entropy = -np.sum(xlogy(probabilities, probabilities))/np.log(2)
    
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    assert Y.size > 0

    probabilities = Y.value_counts() / Y.size
    
    gini_index = 1 - np.sum(probabilities**2)
    
    return gini_index

def MSE(y: pd.Series) -> float:

    # Returns the MSE of the given Series
    assert y.size > 0
    return np.sum((y - y.mean())**2) / y.size


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    assert Y.size > 0
    assert attr.size > 0
    
    if Y.size != attr.size:
        raise ValueError("Both Input series must have the same length.")

    total_entropy = entropy(Y)
    unique_attr = attr.unique()

    for value in unique_attr:
        subset_indices = attr[attr == value].index
        subset_entropy = entropy(Y.iloc[subset_indices])
        subset_weights = attr[attr == value].size / Y.size
        total_entropy -= subset_weights * subset_entropy

    return total_entropy

def gini_IDX(Y: pd.Series, attr: pd.Series) -> float:
    assert Y.size > 0
    assert attr.size > 0
    
    if Y.size != attr.size:
        raise ValueError("Both Input series must have the same length.")

    total_gini = gini_index(Y)
    unique_attr = attr.unique()

    for value in unique_attr:
        subset_indices = attr[attr == value].index
        subset_gini = gini_index(Y.iloc[subset_indices])
        subset_weights = attr[attr == value].size / Y.size
        total_gini -= subset_weights * subset_gini

    return total_gini

def MSE_information_gain(Y: pd.Series, attr: pd.Series) -> float:
    assert Y.size > 0
    assert attr.size > 0

    if Y.size != attr.size:
        raise ValueError("Both Input series must have the same length.")

    total_MSE = MSE(Y)
    unique_attr = attr.unique()

    for value in unique_attr:
        subset_indices = attr[attr == value].index
        subset_MSE = MSE(Y.iloc[subset_indices])
        subset_weights = attr[attr == value].size / Y.size
        total_MSE -= subset_weights * subset_MSE

    return total_MSE

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to whether the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    # entre must be implimented if not real (i.e, discrete)
    assert y.size > 0
    assert len(X) == y.size
    
    if(criterion == 'information_gain' and (not check_ifreal(y))): # information_gain for discrete output (Entropy)
        info = []
        for feature in features:
            info.append(information_gain(y,X[feature]))
        best_attribute = info.index(max(info))
        return features[best_attribute]
    
    elif(criterion == 'information_gain' and check_ifreal(y)): # information_gain for continous output (MSE)
        info = []
        for feature in features:
            info.append(MSE_information_gain(y,X[feature]))
        best_attribute = info.index(max(info))
        return features[best_attribute]
    
    elif(criterion == 'gini_index'): # gini_index for discrete output
        info = []
        for feature in features:
            info.append(gini_IDX(y,X[feature]))
        best_attribute = info.index(max(info))
        return features[best_attribute]


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

        mask = (X[attribute] <= value) 

        X_mask = X[mask]
        y_mask = y[mask]

        Xn_mask = X[~mask]
        yn_mask = y[~mask]

        to_return = ((X_mask, y_mask), (Xn_mask, yn_mask))

        return to_return
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

##  utils for rido
    
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

## utils for riro

def calculate_mse(array):
    return np.sum((array - np.mean(array)) ** 2)


def opt_split(attribute1, output1):
    df = pd.DataFrame({'Attr': attribute1, 'Out': output1})
    dfN = df.sort_values(by='Attr')

    attribute = dfN.iloc[:, 0]
    output = dfN.iloc[:, 1]

    min_mse = np.inf

    if len(attribute) != len(output):
        raise ValueError("Both attribute and output variable series should have the same length!")
    else:
        all_mse = []
        for i in range(len(attribute) - 1):
            split_val = (attribute[i] + attribute[i + 1]) / 2
            left_half = np.array([output[j] for j in range(len(attribute)) if attribute[j] < split_val])
            right_half = np.array([output[j] for j in range(len(attribute)) if attribute[j] >= split_val])
            if len(left_half) == 0:
                left_mse = 0
            else:
                left_mse = calculate_mse(left_half)
            if len(right_half) == 0:
                right_mse = 0
            else:
                right_mse = calculate_mse(right_half)

            total_mse = left_mse + right_mse
            if total_mse < min_mse:
                min_mse = total_mse
                elt = attribute[i]

        split_idx = df[df.iloc[:, 0] == elt].index[0]
        return min_mse, split_idx


def opt_attr(df, target):
    min_mse_in_all_attributes = np.inf
    for col_name in df.columns:
        mse = opt_split(df[col_name], target)[0]
        if mse < min_mse_in_all_attributes:
            min_mse_in_all_attributes = mse
            min_mse_idx = col_name

    opt_split_in_opt_attr = opt_split(df[min_mse_idx], target)[1]
    final_feature = min_mse_idx
    best_split_value = (df[final_feature][opt_split_in_opt_attr] + df[final_feature][opt_split_in_opt_attr + 1]) / 2
    return final_feature, best_split_value
