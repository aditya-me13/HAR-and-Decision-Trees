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


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
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
    if sum([actual == cls for actual in y]) != 0:
        return sum([actual == cls and pred == cls for actual, pred in zip(y, y_hat)])/sum([actual == cls for actual in y])   
    else:
        print(f'No instance of class {cls} was sampled. Hence, recall for class {cls} is not defined.')
        return None 


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
