# contains code for RIRO; still raw

# opt attribute and opt split
import numpy as np
import statistics
import pandas as pd


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
    return int(final_feature), best_split_value


# Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types

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
    if (not check_ifreal(X[attribute])):
        to_return = []
        unique_vals = X[attribute].unique()

        for unique in unique_vals:
            mask = (X[attribute] == unique)
            X_mask = X[mask]
            y_mask = y[mask]
            to_return.append((X_mask, y_mask))

        return to_return

    # Real Input
    # Pass the value to be spliited
    # Only binary split possible
    else:

        mask1 = (X[attribute] <= value)
        mask2 = (X[attribute] > value)
        X_mask = X[mask1]
        y_mask = y[mask1]

        Xn_mask = X[mask2]
        yn_mask = y[mask2]

        to_return = (X_mask, y_mask), (Xn_mask, yn_mask)

        return to_return

        # full=pd.concat[X,y]

        # mask_left = full[attribute].values < full[attribute][value]
        # mask_right = full[attribute].values >= full[attribute][value]
        # df_new_left = X.loc[mask_left]
        # df_new_right = X.loc[mask_right]
        # X_left=df_new_left[:,:-1]
        # y_left=df_new_left[:,-1]
        # X_right=df_new_right[:,:-1]
        # y_right=df_new_right[:,-1]

        # return ((X_left, y_left), (X_right, y_right))

        # df = pd.concat([X,y], axis = 1)

        # left = df[df[attribute] <= value]
        # right = df[df[attribute] > value]

        # return((left.iloc[:,:-1], left.iloc[:,-1]), (right.iloc[:,:-1], right.iloc[:,-1]))


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
                node_.decision = y.mean()
                print(node_.depth)
                print(node_.decision)
                return
            if len(features) == 0:
                node_.decision = y.mean()
                print(node_.depth)
                print(node_.decision)
                return
            elif max_depth == 0:
                node_.decision = y.mean()
                print(node_.depth)
                print(node_.decision)
                return
            else:
                opt_attribute, split_val = opt_attr(X, y)
                # print(opt_attribute, split_val)
                node_.to_split = opt_attribute
                node_.split_val = split_val
                t, r = split_data(X, y, opt_attribute, split_val)
                left_df_X, left_df_y = dfconvertor(t)
                right_df_X, right_df_y = dfconvertor(r)
                # print("left df")
                # display(left_df_X)
                # display(left_df_y)
                # print()
                # print('right df')
                # display(right_df_X)
                # display(right_df_y)
                # print()

                new_node = Node(depth=node_.depth + 1)
                node_.left_ptr = new_node
                # print('depth of node, left side')
                # print(node_.depth)
                # print()
                features_ = [feature for feature in features if feature != opt_attribute]
                make_tree(left_df_X, left_df_y, features_, criterion, max_depth - 1, new_node)

                new_node_ = Node(depth=node_.depth + 1)
                node_.right_ptr = new_node_
                # print('depth of node, right side')
                # print(node_.depth)
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

        def predict_from_node(dataFrame, root, max_depth):
            ## Do check out for len(dataFrame.columns) == 0 condition
            if (root.decision is not None):
                # print('yes')
                return root.decision

            split_attribute = root.to_split
            split_value = root.split_val
            # print(split_value)

            if dataFrame[split_attribute][0] <= split_value:
                # print(root.left_ptr.depth)
                # print()
                return predict_from_node(dataFrame, root.left_ptr, max_depth)

            else:
                # print(root.right_ptr.depth)
                # print()
                return predict_from_node(dataFrame, root.right_ptr, max_depth)

        def predict_(X: pd.DataFrame, tree_root, max_depth):
            y = []
            for i in range(X.shape[0]):
                df = pd.DataFrame(X.iloc[i, :]).T
                df.reset_index(inplace=True, drop=True)
                y.append(predict_from_node(df, tree_root, max_depth))
            return pd.Series(y)

        return predict_(X, self.Root, self.max_depth)
