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
from tree.utils import *
from metrics import *
np.random.seed(42)

@dataclass

class Next:
    def __init__(self,name, ptr = None):
        self.ptr = ptr
        self.name = name

class Node:
    def __init__(self, to_split=None, split_val=0.0, depth=float('inf'), decision = None):
        self.to_split = to_split
        self.split_val = split_val
        self.depth = depth
        self.decision = decision
        self.left_ptr = None
        self.right_ptr = None
        self.nexts = []

    def add_in_nexts(self,name, ptr=None):
        lst = self.nexts
        lst.append(Next(name = name,ptr = ptr))
        self.nexts = lst

class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int  

    def __init__(self, criterion='information_gain', max_depth=4):
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
        
        def make_tree(X: pd.DataFrame, y:pd.Series, features:list, criterion:str, max_depth:int, node_, value):
            
            # same condition for all cases 
            if(y.nunique() == 1):
                node_.decision = y.mode()[0]
                return

            # for discrete outputs
            elif(len(features) == 0 and (not check_ifreal(y))):
                node_.decision = y.mode()[0]
                return
            elif (node_.depth > max_depth and (not check_ifreal(y))):
                node_.decision = y.mode()[0] 
                return

            # for real outputs
            elif(len(features) == 0 and (check_ifreal(y))):
                node_.decision = y.mean() 
                return    
            elif (node_.depth > max_depth and (check_ifreal(y))):
                node_.decision = y.mean() 
                return         
            
            else:

                #  for discrete inputs
                if (not check_ifreal(X.iloc[:,0])):
                    opt_attr_ = opt_split_attribute(X, y, criterion, pd.Series(features))
                    node_.to_split = opt_attr_
                    for element in split_data(X, y, opt_attr_, 0):
                        Xi, yi = dfconvertor(element) 
                        node_.add_in_nexts(name = Xi[opt_attr_].unique()[0])
                        new_node = Node(depth = node_.depth + 1, split_val=value)
                        node_.nexts[-1].ptr = new_node 
                        features_ = [feature for feature in features if feature != opt_attr_]
                        make_tree(Xi, yi, features_, criterion, max_depth, new_node, value)
                    return 
                
                # for real inputs
                else:
                    # for RIRO
                    if check_ifreal(y):
                        opt_attribute, split_val = opt_attr(X, y)

                    #for RIDO    
                    else:
                        optimal_attribute = find_optimal_attribute(df_to_array(X), np.array(y))
                        opt_attribute = X.columns[optimal_attribute.feature_index]
                        split_val = optimal_attribute.split_value

                    # for real input
                    node_.to_split = opt_attribute
                    node_.split_val = split_val
                    t, r = split_data(X, y, opt_attribute, split_val)
                    left_df_X, left_df_y = dfconvertor(t)
                    right_df_X, right_df_y = dfconvertor(r)

                    new_node = Node(depth=node_.depth + 1)
                    node_.left_ptr = new_node
    
                    make_tree(left_df_X, left_df_y, features, criterion, max_depth, new_node, value)

                    new_node_ = Node(depth=node_.depth + 1)
                    node_.right_ptr = new_node_

                    make_tree(right_df_X, right_df_y, features, criterion, max_depth, new_node_, value)
                return                        
            
        self.Root = Node(depth = 0, split_val = y.mean() if check_ifreal(y) else y.mode()[0])   
        make_tree(X, y, list(X.columns), self.criterion, self.max_depth, self.Root, self.Root.split_val)
        return None    
    
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        # for discrete inputs
        if (not check_ifreal(X.iloc[:,0])):
            def predict_from_node(X: pd.DataFrame, node):
                if (node.decision is not None):
                    return node.decision
                else:
                    split_attribute = node.to_split
                    split_attribute_val = X[split_attribute]
                    flag = 0
                    for i in node.nexts:
                        if i.name == split_attribute_val:
                            flag = 1
                            return predict_from_node(X,i.ptr)
                    if not flag:
                        return node.split_val

            def predict_(X: pd.DataFrame, Tree_Root):
                y = []
                for i in range(X.shape[0]):
                    y.append(predict_from_node(X.iloc[i,:], Tree_Root))
                return pd.Series(y)


        else:
            # for real inputs
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
            if (node_.left_ptr is not None or node_.right_ptr is not None and node_.nexts == []):
                return
            else:
                if (node_.to_split is None):
                    print(node_.decision)
                    return None
                else: # node_decision is None
                    print('?(',node_.to_split,')', sep='')
                    for element in node_.nexts:
                        print("   "*depth, element.name, ': ',end = '', sep = '')
                        plot_(element.ptr, depth+1)                

        def _plot_(node_, depth):
            if (node_.left_ptr == None and node_.right_ptr == None and node_.nexts != [] ):
                return
            else:
                if (node_.decision is not None):
                    print("   "*depth,node_.decision)
                    return None
                else: # node_decision is None
                    print("   "*depth,'?(', node_.to_split, '<=', node_.split_val, ')', sep='')
                    _plot_(node_.left_ptr, depth+1)

                    print("   "*depth,'?(', node_.to_split, '>', node_.split_val, ')', sep='')
                    _plot_(node_.right_ptr, depth+1)                        
        plot_(self.Root, 0)
        _plot_(self.Root,0)
        return None 